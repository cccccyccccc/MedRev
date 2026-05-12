# MedRev 医学影像标注审核系统

MedRev 是一个本地部署的医学影像审核工具，用于医生快速审核两类任务：

- 页面一：困难样本 GT 检查，医生判断原始 GT 是否正确。
- 页面二：无标注图片伪标签检查，医生判断模型伪标签是否正确。

系统当前适配真实生产数据目录 `Test/`。管理员看到和选择的是真实数据根目录 `Test`，系统内部会自动整理到 `test_data/整理后标注目录/<器官>/` 后再生成任务。

## 安装与启动

建议使用 Python 3.10+。

```bash
pip install -r requirements.txt
python backend/app.py
```

默认启动地址：

```text
http://127.0.0.1:5001
```

如果需要局域网内其他设备访问，可以把 `backend/app.py` 最后的启动参数改为 `app.run(host="0.0.0.0", port=5001)`，然后用服务器 IP 访问。

## 默认账号

账号配置在 `backend/users.json`。

| 角色 | 用户名 | 密码 |
|---|---|---|
| 管理员 | `admin` | `adminpass` |
| 医生 | `doctor_1` | `password1` |
| 医生 | `doctor_2` | `password2` |

登录入口：

```text
http://127.0.0.1:5001/login
```

测试时建议用两个浏览器或一个普通窗口加一个无痕窗口分别登录：

- 浏览器 A 登录管理员账号，打开 `/admin` 或 `/admin/settings`。
- 浏览器 B 登录医生账号，打开 `/doctor`。

这样可以避免同一个浏览器 Session 相互覆盖，方便一边分配任务、一边模拟医生审核。

当前页面没有做实时推送同步。管理员分配任务后，医生需要刷新 `/doctor` 或重新点击“取下一条”才能看到最新分配；医生提交审核后，管理员需要刷新 `/admin` 或 `/admin/results` 才能看到最新审核进度和结果统计。

## 常用路由

| 路由 | 用途 |
|---|---|
| `/login` | 登录页 |
| `/logout` | 退出登录 |
| `/doctor` | 医生审核页 |
| `/admin` | 管理员主页，查看任务池、分配任务、导出结果 |
| `/admin/settings` | 管理员设置页，生成新轮次、选择数据目录 |
| `/admin/results` | 管理员结果页 |
| `/task/<task_id>` | 查看单条任务 JSON |
| `/task/<task_id>/asset/current` | 当前审核图片 |
| `/task/<task_id>/asset/report` | 病例报告/关联检查图 |
| `/task/<task_id>/asset/related?index=0` | 同病例其他图片 |
| `/render?task_id=<task_id>&show=gt` | 渲染 GT 展示图 |
| `/render?task_id=<task_id>&show=pred` | 渲染预测/伪标签展示图 |

## 数据目录逻辑

当前项目有两个容易混淆的目录：

```text
Test/       真实生产数据源，管理员页面默认显示这个
test_data/  系统内部工程化缓存目录，由适配脚本生成
```

生产数据目录示例：

```text
Test/
  GT.json
  pred.json
  organ.json
  raw_subset/
```

系统启动或管理员生成新轮次时，如果发现 `data_root=Test`，会先自动执行生产数据适配，生成：

```text
test_data/整理后标注目录/肾脏/
  GT.json
  pred.json
  organ.json
  case_metadata.json
```

然后 `scripts/generate_tasks.py` 从该工程化目录生成任务。

## 当前默认配置

配置文件是 `config.json`。

关键字段：

```json
{
  "run_id": "test_run_001",
  "data_root": "Test",
  "prepared_data_root": "test_data",
  "bootstrap_organ": "肾脏",
  "conf_threshold": 0.7,
  "iou_threshold": 0.5,
  "hard_sample_mode": "strict",
  "reset_outputs_on_start": true
}
```

说明：

- `data_root`：真实数据源，当前默认 `Test`。
- `prepared_data_root`：系统整理后的缓存目录，当前默认 `test_data`。
- `run_id`：当前审核轮次。
- `reset_outputs_on_start`：为 `true` 时，启动后会自动用默认数据重新生成任务并清空当前轮次输出。
- `hard_sample_mode=strict`：页面一按 GT 与预测的完整匹配结果筛选困难样本；当前 strict 逻辑不按预测置信度过滤页面一候选框。

当前仓库默认是测试配置：

```json
"reset_outputs_on_start": true
```

这意味着每次重新启动后端时，系统会重置当前轮次的审核输出和任务分配，并基于默认 `Test` 数据重新生成任务。测试时这很方便；如果需要保留审核结果，请改为：

```json
"reset_outputs_on_start": false
```

当前真实数据默认生成：

```text
页面一任务：1 条
页面二任务：113 条
```

## 医生操作流程

1. 访问 `/login`，使用医生账号登录。
2. 登录后进入 `/doctor`。
3. 在顶部选择任务类型：
   - 页面一：困难样本 GT 检查
   - 页面二：无标注伪标签检查
4. 点击“取下一条”加载任务。
5. 查看病例报告图、当前图像、GT/预测展示图、同病例其他图片。
6. 页面一只需选择：
   - `GT 正确`
   - `GT 有误`
7. 页面二选择：
   - `伪标签正确`
   - 或选择固定错误类型后提交错误
8. 保存成功后自动进入下一条任务。

如果管理员刚刚分配了新任务，医生页面需要刷新或重新点击“取下一条”获取最新任务。

页面异常处理：

- 病例报告图缺失：显示“暂无病例报告图片”。
- 当前图片加载失败：显示加载失败并可重试。
- 同病例图片为空：显示“暂无其他图片”。
- GT 或预测信息缺失：显示对应缺失提示。
- 保存失败：保留当前页面并提示重试。

## 管理员操作流程

1. 访问 `/login`，使用 `admin / adminpass` 登录。
2. 进入 `/admin` 查看任务池、任务统计、医生分配情况。
3. 进入 `/admin/settings` 生成新轮次：
   - 选择器官，例如 `肾脏`
   - 填写模型版本
   - 填写数据版本
   - 数据根目录默认 `Test`
   - 点击“生成新轮次”
4. 回到 `/admin`：
   - 查看页面一和页面二任务
   - 选择任务分配给医生
   - 查看审核进度
5. 点击导出，或由系统自动导出审核结果。

医生提交审核后，管理员页面需要刷新才能看到最新进度、任务状态和导出统计。

## 输出结果

审核和导出结果保存在：

```text
review_outputs/<run_id>/
```

常见文件：

```text
raw_reviews.jsonl
gt_issue_list.jsonl
accepted_pseudo_labels.jsonl
accepted_pseudo_labels.coco.json
pseudo_label_error_list.jsonl
summary.json
assignments.json
```

系统同时使用 SQLite 保存审核记录和任务分配，数据库文件为：

```text
medrev.db
```

## 常用脚本

手动适配真实生产数据：

```bash
python scripts/adapt_production_test_data.py --source-json-dir Test --source-root Test/raw_subset --output-root test_data --organ 肾脏
```

手动生成任务：

```bash
python scripts/generate_tasks.py --run-id test_run_001 --organ 肾脏 --data-root test_data --conf-threshold 0.7 --iou-threshold 0.5 --hard-sample-mode strict
```

手动导出审核结果：

```bash
python scripts/export_review_results.py --run-id test_run_001
```

清空当前审核和分配记录：

```bash
python scripts/reset_reviews.py
```

## run_id 与 task_id

- `run_id` 表示一轮任务生成和审核批次，例如 `test_run_001`。
- `task_id` 表示该轮中的一条具体任务，包含任务类型和图片路径信息。

示例：

```text
test_run_001_hard_肾囊肿_50岁以上_女116_肾脏_p0110_肾脏_p0110_011_081241.jpg
test_run_001_pseudo_肾囊肿_50岁以上_女116_肾脏_p0110_肾脏_p0110_003_081240.jpg
```

审核结果通过 `run_id + task_id + organ + image_name` 追踪，支持后续模型迭代闭环。
