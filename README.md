# dsh-plugin-novelai

> **像聊天一样生图，把想象直接说成画面。** 告别枯燥的提示词工程——描述你脑海里的画面，AI 自动按专业规则写好 NovelAI V4.5 提示词并出图，不用再手搓 Danbooru tags。

NovelAI Diffusion V4.5 Full 生图插件 for [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness)（DSH）。

在对话里描述你想要的图片，AI 会按内置的 NovelAI 提示词设计规则（照搬自 ChatChatBar 的 `NOVELAI_IMAGE_PROMPT_SYSTEM`）写好提示词，然后调用 `nai_generate_image` 工具真正生成。生成的 PNG 保存到本地文件，工具结果只含文本，纯文本模型适配器不会收到图片内容块。

## 安装

```sh
# 从 GitHub（生产推荐）
dsh plugin --profile web add github:ysysz0343-web/dsh-plugin-novelai

# 从 npm
dsh plugin --profile web add @scope/dsh-plugin-novelai

# 本地开发：用 file: 前缀（复制安装）。裸路径会被 pnpm 当作 link: 软链，
# 软链跟随真实路径，导致 @deepseek-ai/* peer 依赖无法通过 profiles/node_modules 回退解析。
dsh plugin --profile web add file:./path/to/dsh-plugin-novelai
```

> 仓库需公开；`dsh-plugin` topic 是第三方插件目录（deepseek-harness-plugin.com）收录的要求，非官方硬性要求。
> 本插件是纯 JavaScript（无 TypeScript 构建步骤），git 安装直接可用、**无需任何构建授权**（不同于需要 `prepare` 脚本的 TypeScript 插件）。
> 安装后部分版本需要重启 Harness（`dsh web`）才生效。

验证：

```sh
dsh plugins list
```

## 配置 NovelAI Token

三种方式（优先级从高到低）：

1. 工具调用时传 `token` 参数；
2. `cordis.patch.yml` 中 `config.token`；
3. 环境变量 `NOVELAI_API_KEY`（或在 `$DSH_HOME/.credentials.yaml` 配置该 key）。

## 使用方法

配置好 Token 后，直接在对话里说，例如：

> 帮我生成一张 nai4.5f 图：雨夜里银发少女撑着红伞站在霓虹窄巷，湿漉漉的裙摆，侧后方视角

AI 会设计好 `base_caption` / `characters` / `size_preset`，调用 `nai_generate_image`，图片保存到会话工作区的 `nai-images` 目录（默认），工具结果里会返回文件路径。

### 工具参数

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `base_caption` | string (必填) | 整体画面 caption，以人数开头（1girl / 2girls / 1boy, 1girl） |
| `characters` | array | 每个角色的 caption（外观标签，最多 6 个），可选 `center` 坐标 |
| `size_preset` | string | `PORTRAIT`=832x1216 / `SQUARE`=1024x1024 / `HORIZONTAL`=1216x832 |
| `negative_prompt` | string | 负向提示词，留空用内置默认 |
| `seed` | number | 随机种子，留空随机 |
| `batch_size` | number | 生成张数 1-4，默认 1 |
| `token` | string | 可选，临时 Token 覆盖 |
| `proxy` | string | 可选，HTTP 代理（默认 `http://127.0.0.1:2080`；`none`/空 = 直连） |

## 实现说明

- 端点 `POST https://image.novelai.net/ai/generate-image-stream`，模型 `nai-diffusion-4-5-full`，请求体逐字段对齐 ChatChatBar 的 V4 实现（`params_version:3`、28 steps、scale 8、`k_euler_ancestral`、Karras、`v4_prompt`/`v4_negative_prompt` 等）。
- 响应是 MessagePack 二进制流（4 字节大端长度前缀帧），插件内置纯 JS 解码器解析 `intermediate` / `final` / `error`。
- 请求走 Windows 自带 `curl.exe`（`-x` 支持代理），因为 DSH 的 `web.fetch` 只支持 GET 文本。
- Token 不落盘；插件进程内不持久化任何凭据。

## 安全提示

插件会调用外部 API（`image.novelai.net`）并使用你的 NovelAI Token。安装来自 GitHub 的插件前请先检查源码。

## License

MIT
