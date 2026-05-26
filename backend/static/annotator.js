/**
 * Canvas-based bounding-box annotator.
 * Stores all coordinates in original image space; renders scaled to fit.
 */
class Annotator {
  static CATEGORIES = [
    { id: 0, name: '肾错构瘤', color: '#FF6B6B' },
    { id: 1, name: '肾积水', color: '#4ECDC4' },
    { id: 2, name: '肾结石', color: '#45B7D1' },
    { id: 3, name: '肾囊肿', color: '#96CEB4' },
    { id: 4, name: '肾实质弥漫性病变', color: '#DDA0DD' },
    { id: 5, name: '肾脏恶性肿瘤', color: '#FFA07A' },
  ];

  constructor(container, options = {}) {
    this.container = container;
    this.onChange = options.onChange || (() => {});
    this.onSelection = options.onSelection || (() => {});

    this.annotations = [];
    this._nextId = 1;
    this.mode = 'select';
    this.selectedId = null;
    this.activeLabel = Annotator.CATEGORIES[0];
    this.image = null;
    this.imgW = 0;
    this.imgH = 0;
    this.scale = 1;
    this.ox = 0;
    this.oy = 0;

    // interaction
    this._drawing = false;
    this._dx0 = 0; this._dy0 = 0; this._dx1 = 0; this._dy1 = 0;
    this._dragging = false;
    this._dragOx = 0; this._dragOy = 0;
    this._dragOrig = null;
    this._resizing = false;
    this._resizeH = null;
    this._resizeOx = 0; this._resizeOy = 0;
    this._resizeOrig = null;
    this._hoveredId = null;

    this._init();
  }

  _init() {
    this.canvas = document.createElement('canvas');
    this.canvas.id = 'annotationCanvas';
    this.container.appendChild(this.canvas);
    this.ctx = this.canvas.getContext('2d');

    this.canvas.addEventListener('mousedown', this._onDown.bind(this));
    this.canvas.addEventListener('mousemove', this._onMove.bind(this));
    window.addEventListener('mouseup', this._onUp.bind(this));
    this.canvas.addEventListener('dblclick', this._onDbl.bind(this));
    this._keyFn = this._onKey.bind(this);
    document.addEventListener('keydown', this._keyFn);

    this._ro = new ResizeObserver(() => this._syncSize());
    this._ro.observe(this.container);
    this._syncSize();
  }

  _syncSize() {
    const r = this.container.getBoundingClientRect();
    const w = r.width || 800;
    const h = r.height || 600;
    const dpr = window.devicePixelRatio || 1;
    if (this.canvas.width === Math.round(w * dpr) && this.canvas.height === Math.round(h * dpr)) return;
    this.canvas.width = Math.round(w * dpr);
    this.canvas.height = Math.round(h * dpr);
    this.canvas.style.width = w + 'px';
    this.canvas.style.height = h + 'px';
    this._recalcTransform();
    this.render();
  }

  _recalcTransform() {
    if (!this.imgW || !this.imgH) return;
    const cw = this.canvas.width / (window.devicePixelRatio || 1);
    const ch = this.canvas.height / (window.devicePixelRatio || 1);
    this.scale = Math.min(cw / this.imgW, ch / this.imgH);
    this.ox = (cw - this.imgW * this.scale) / 2;
    this.oy = (ch - this.imgH * this.scale) / 2;
  }

  /* ---- coordinate helpers ---- */
  _img2c(imgX, imgY) {
    return { x: imgX * this.scale + this.ox, y: imgY * this.scale + this.oy };
  }
  _c2img(canvasX, canvasY) {
    const dpr = window.devicePixelRatio || 1;
    const cx = canvasX * dpr;
    const cy = canvasY * dpr;
    const cw = this.canvas.width;
    const ch = this.canvas.height;
    const sx = cw / (this.imgW * this.scale || 1);
    const sy = ch / (this.imgH * this.scale || 1);
    return {
      x: (cx / cw * (this.imgW * this.scale) - this.ox * sx) / this.scale,
      y: (cy / ch * (this.imgH * this.scale) - this.oy * sy) / this.scale,
    };
  }
  _c2imgSimple(cx, cy) {
    const dpr = window.devicePixelRatio || 1;
    return {
      x: (cx * dpr - this.ox * dpr) / (this.scale * dpr),
      y: (cy * dpr - this.oy * dpr) / (this.scale * dpr),
    };
  }
  _clampImg(x, y) {
    return { x: Math.max(0, Math.min(this.imgW, x)), y: Math.max(0, Math.min(this.imgH, y)) };
  }

  /* ---- public API ---- */

  loadImage(src, imgW, imgH) {
    this.imgW = imgW;
    this.imgH = imgH;
    this.annotations = [];
    this._nextId = 1;
    this.selectedId = null;
    this.image = new Image();
    this.image.onload = () => {
      if (!this.imgW) { this.imgW = this.image.naturalWidth; this.imgH = this.image.naturalHeight; }
      this._recalcTransform();
      this.render();
    };
    this.image.src = src;
    this._recalcTransform();
    this.render();
  }

  loadAnnotations(list) {
    this.annotations = (list || []).map((a, i) => ({
      id: this._nextId++,
      bbox: [...a.bbox],
      category: a.category || this.activeLabel.name,
      category_id: a.category_id != null ? a.category_id : this.activeLabel.id,
      source: a.source || 'loaded',
      confidence: a.confidence,
    }));
    this.selectedId = null;
    this.render();
    this._notify();
  }

  getAnnotations() {
    return this.annotations.map((a) => ({
      id: a.id,
      bbox: [...a.bbox],
      category: a.category,
      category_id: a.category_id,
      source: a.source,
    }));
  }

  getCocoJson(fileName) {
    const images = [{
      id: 1,
      file_name: fileName || 'image.jpg',
      width: this.imgW,
      height: this.imgH,
    }];
    const annotations = this.annotations
      .filter((a) => a.source === 'user' || a.source === 'user_edited')
      .map((a, i) => {
        const [x1, y1, x2, y2] = a.bbox;
        const w = x2 - x1;
        const h = y2 - y1;
        return {
          id: i + 1,
          image_id: 1,
          category_id: a.category_id,
          bbox: [Math.round(x1), Math.round(y1), Math.round(w), Math.round(h)],
          area: Math.round(w * h),
          iscrowd: 0,
        };
      });
    const categories = Annotator.CATEGORIES.map((c) => ({
      id: c.id,
      name: c.name,
    }));
    return { images, annotations, categories };
  }

  setActiveLabel(catId) {
    const c = Annotator.CATEGORIES.find((x) => x.id === catId);
    if (c) this.activeLabel = c;
  }

  setMode(m) {
    this.mode = m;
    this.canvas.style.cursor = m === 'draw' ? 'crosshair' : 'default';
    this.render();
  }

  deleteSelected() {
    if (this.selectedId == null) return;
    this.annotations = this.annotations.filter((a) => a.id !== this.selectedId);
    this.selectedId = null;
    this.render();
    this._notify();
  }

  changeSelectedLabel(catId) {
    if (this.selectedId == null) return;
    const a = this.annotations.find((x) => x.id === this.selectedId);
    if (!a) return;
    const c = Annotator.CATEGORIES.find((x) => x.id === catId);
    if (!c) return;
    a.category = c.name;
    a.category_id = c.id;
    if (a.source === 'loaded') a.source = 'user_edited';
    this.render();
    this._notify();
  }

  destroy() {
    this._ro.disconnect();
    document.removeEventListener('keydown', this._keyFn);
    this.canvas.remove();
  }

  /* ---- rendering ---- */

  render() {
    const c = this.ctx;
    const cw = this.canvas.width;
    const ch = this.canvas.height;
    const dpr = window.devicePixelRatio || 1;
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, cw / dpr, ch / dpr);

    // image
    if (this.image && this.image.complete) {
      c.drawImage(this.image, this.ox, this.oy, this.imgW * this.scale, this.imgH * this.scale);
    }

    // boxes
    this.annotations.forEach((a) => this._drawBox(a, a.id === this.selectedId, a.id === this._hoveredId));

    // drawing preview
    if (this._drawing) {
      const p0 = this._img2c(this._dx0, this._dy0);
      const p1 = this._img2c(this._dx1, this._dy1);
      c.strokeStyle = this.activeLabel.color;
      c.lineWidth = 2;
      c.setLineDash([6, 3]);
      c.strokeRect(p0.x, p0.y, p1.x - p0.x, p1.y - p0.y);
      c.setLineDash([]);
    }
  }

  _drawBox(a, selected, hovered) {
    const c = this.ctx;
    const p0 = this._img2c(a.bbox[0], a.bbox[1]);
    const p2 = this._img2c(a.bbox[2], a.bbox[3]);
    const color = Annotator.CATEGORIES.find((x) => x.id === a.category_id)?.color || '#999';

    const lw = selected ? 3 : hovered ? 2 : 1.5;
    c.strokeStyle = selected ? '#fff' : color;
    c.lineWidth = lw;
    c.strokeRect(p0.x, p0.y, p2.x - p0.x, p2.y - p0.y);

    // fill with low opacity
    c.fillStyle = color + '1a';
    c.fillRect(p0.x, p0.y, p2.x - p0.x, p2.y - p0.y);

    // label
    const label = a.category;
    c.font = '12px "Segoe UI","PingFang SC","Microsoft YaHei",sans-serif';
    const tm = c.measureText(label);
    const tw = tm.width;
    const th = 16;
    const lx = p0.x;
    const ly = p0.y - th - 2 > 0 ? p0.y - th - 2 : p0.y;
    c.fillStyle = color;
    c.fillRect(lx, ly, tw + 8, th);
    c.fillStyle = '#fff';
    c.fillText(label, lx + 4, ly + 12);

    // handles when selected
    if (selected) {
      this._drawHandles(p0, p2, color);
    }
  }

  _drawHandles(p0, p2, color) {
    const c = this.ctx;
    const s = 6;
    const pts = [
      { x: p0.x, y: p0.y }, { x: (p0.x + p2.x) / 2, y: p0.y }, { x: p2.x, y: p0.y },
      { x: p2.x, y: (p0.y + p2.y) / 2 }, { x: p2.x, y: p2.y },
      { x: (p0.x + p2.x) / 2, y: p2.y }, { x: p0.x, y: p2.y }, { x: p0.x, y: (p0.y + p2.y) / 2 },
    ];
    pts.forEach((pt) => {
      c.fillStyle = '#fff';
      c.fillRect(pt.x - s / 2, pt.y - s / 2, s, s);
      c.strokeStyle = color;
      c.lineWidth = 1.5;
      c.strokeRect(pt.x - s / 2, pt.y - s / 2, s, s);
    });
  }

  /* ---- hit testing ---- */

  _hitTestAnn(cx, cy) {
    const pt = this._c2imgSimple(cx, cy);
    // search topmost (last drawn) first
    for (let i = this.annotations.length - 1; i >= 0; i--) {
      const b = this.annotations[i].bbox;
      if (pt.x >= b[0] && pt.x <= b[2] && pt.y >= b[1] && pt.y <= b[3]) return this.annotations[i];
    }
    return null;
  }

  _hitHandle(cx, cy) {
    if (this.selectedId == null) return null;
    const a = this.annotations.find((x) => x.id === this.selectedId);
    if (!a) return null;
    const p0 = this._img2c(a.bbox[0], a.bbox[1]);
    const p2 = this._img2c(a.bbox[2], a.bbox[3]);
    const s = 10;
    const handles = [
      { n: 'nw', x: p0.x, y: p0.y }, { n: 'n', x: (p0.x + p2.x) / 2, y: p0.y },
      { n: 'ne', x: p2.x, y: p0.y }, { n: 'e', x: p2.x, y: (p0.y + p2.y) / 2 },
      { n: 'se', x: p2.x, y: p2.y }, { n: 's', x: (p0.x + p2.x) / 2, y: p2.y },
      { n: 'sw', x: p0.x, y: p2.y }, { n: 'w', x: p0.x, y: (p0.y + p2.y) / 2 },
    ];
    for (const h of handles) {
      if (Math.abs(cx - h.x) < s && Math.abs(cy - h.y) < s) return h.n;
    }
    return null;
  }

  _handleCursor(cx, cy) {
    if (this.mode === 'draw') { this.canvas.style.cursor = 'crosshair'; return; }
    if (this._dragging || this._resizing) return;
    const h = this._hitHandle(cx, cy);
    if (h) {
      const map = { nw: 'nwse-resize', se: 'nwse-resize', ne: 'nesw-resize', sw: 'nesw-resize', n: 'ns-resize', s: 'ns-resize', e: 'ew-resize', w: 'ew-resize' };
      this.canvas.style.cursor = map[h] || 'default';
      return;
    }
    const hit = this._hitTestAnn(cx, cy);
    this._hoveredId = hit ? hit.id : null;
    this.canvas.style.cursor = hit ? 'move' : 'default';
  }

  /* ---- events ---- */

  _onDown(e) {
    const rect = this.canvas.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;

    if (this.mode === 'draw') {
      const pt = this._c2imgSimple(cx, cy);
      this._drawing = true;
      this._dx0 = pt.x; this._dy0 = pt.y;
      this._dx1 = pt.x; this._dy1 = pt.y;
      this.render();
      return;
    }

    // select mode
    const handle = this._hitHandle(cx, cy);
    if (handle && this.selectedId != null) {
      this._resizing = true;
      this._resizeH = handle;
      this._resizeOx = cx; this._resizeOy = cy;
      const a = this.annotations.find((x) => x.id === this.selectedId);
      this._resizeOrig = a ? [...a.bbox] : null;
      return;
    }

    const hit = this._hitTestAnn(cx, cy);
    if (hit) {
      this.selectedId = hit.id;
      this._dragging = true;
      this._dragOx = cx; this._dragOy = cy;
      this._dragOrig = [...hit.bbox];
      this._onSelection(hit);
      this.render();
      return;
    }

    this.selectedId = null;
    this._onSelection(null);
    this.render();
  }

  _onMove(e) {
    const rect = this.canvas.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    this._handleCursor(cx, cy);

    if (this._drawing) {
      const pt = this._c2imgSimple(cx, cy);
      const c = this._clampImg(pt.x, pt.y);
      this._dx1 = c.x; this._dy1 = c.y;
      this.render();
      return;
    }

    if (this._dragging && this._dragOrig) {
      const pt0 = this._c2imgSimple(this._dragOx, this._dragOy);
      const pt1 = this._c2imgSimple(cx, cy);
      const dx = pt1.x - pt0.x;
      const dy = pt1.y - pt0.y;
      const a = this.annotations.find((x) => x.id === this.selectedId);
      if (a) {
        a.bbox = [
          Math.max(0, this._dragOrig[0] + dx),
          Math.max(0, this._dragOrig[1] + dy),
          Math.min(this.imgW, this._dragOrig[2] + dx),
          Math.min(this.imgH, this._dragOrig[3] + dy),
        ];
        this.render();
      }
      return;
    }

    if (this._resizing && this._resizeOrig) {
      const pt = this._c2imgSimple(cx, cy);
      const c2 = this._clampImg(pt.x, pt.y);
      const a = this.annotations.find((x) => x.id === this.selectedId);
      if (a) {
        let [x1, y1, x2, y2] = this._resizeOrig;
        const h = this._resizeH;
        if (h.includes('w')) x1 = c2.x;
        if (h.includes('e')) x2 = c2.x;
        if (h.includes('n')) y1 = c2.y;
        if (h.includes('s')) y2 = c2.y;
        if (x2 > x1 && y2 > y1) { a.bbox = [x1, y1, x2, y2]; }
        this.render();
      }
      return;
    }
  }

  _onUp(e) {
    if (this._drawing) {
      this._drawing = false;
      const x1 = Math.min(this._dx0, this._dx1);
      const y1 = Math.min(this._dy0, this._dy1);
      const x2 = Math.max(this._dx0, this._dx1);
      const y2 = Math.max(this._dy0, this._dy1);
      if (x2 - x1 > 5 && y2 - y1 > 5) {
        this.annotations.push({
          id: this._nextId++,
          bbox: [x1, y1, x2, y2],
          category: this.activeLabel.name,
          category_id: this.activeLabel.id,
          source: 'user',
        });
        this.selectedId = null;
        this._notify();
      }
      this.render();
    }
    if (this._dragging) {
      this._dragging = false;
      this._dragOrig = null;
      const a = this.annotations.find((x) => x.id === this.selectedId);
      if (a && a.source === 'loaded') a.source = 'user_edited';
      this._notify();
    }
    if (this._resizing) {
      this._resizing = false;
      this._resizeH = null;
      this._resizeOrig = null;
      const a = this.annotations.find((x) => x.id === this.selectedId);
      if (a && a.source === 'loaded') a.source = 'user_edited';
      this._notify();
    }
  }

  _onDbl(e) {
    const rect = this.canvas.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    const hit = this._hitTestAnn(cx, cy);
    if (hit) {
      this.selectedId = hit.id;
      this._onSelection(hit);
      this.render();
    }
  }

  _onKey(e) {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return;
    if ((e.key === 'Delete' || e.key === 'd' || e.key === 'D') && this.selectedId != null) {
      e.preventDefault();
      this.deleteSelected();
    }
    if (e.key === 'Escape') {
      this.selectedId = null;
      this._onSelection(null);
      this.render();
    }
  }

  _notify() {
    this.onChange(this.getAnnotations());
  }

  _onSelection(a) {
    this.onSelection(a);
  }
}
