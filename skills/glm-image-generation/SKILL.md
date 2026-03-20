---
name: glm-image-generation
description:
  Official skill for generating high-quality images from text prompts using ZhiPu GLM-Image API.
  Excellent at scientific illustrations, high-quality portraits, social media graphics, and commercial posters.
  Supports multiple aspect ratios, HD quality, and watermark control.
  Use this skill when the user wants to generate images, create AI art, text-to-image,
  or convert text descriptions into visual content.
metadata:
  openclaw:
    requires:
      env:
        - ZHIPU_API_KEY
      bins:
        - python
    primaryEnv: ZHIPU_API_KEY
    emoji: "🎨"
    homepage: https://github.com/zai-org/GLM-Image/tree/main/skills/glm-image-generation
---

# GLM-Image Generation Skill / GLM-Image 图片生成技能

Generate high-quality images from text prompts using the ZhiPu GLM-Image API.

## When to Use / 使用场景

- Generate images from text descriptions / 从文字描述生成图片
- Create AI art, illustrations, or concept art / 创作 AI 艺术、插画或概念图
- User mentions "生图", "文生图", "AI 绘画", "generate image", "text-to-image", "create image"
- User provides a prompt and wants to see it visualized / 用户提供描述并想看到可视化效果

## Key Features / 核心特性

- **High-quality generation**: HD mode produces more detailed, refined images (~20s)
- **Multiple aspect ratios**: Square, portrait, landscape formats supported
- **GLM-Image model**: Latest model with improved understanding and quality
- **Excellent at**: Scientific illustrations (科普插画), high-quality portraits (高质量人像), social media graphics (社交媒体图文), commercial posters (商业海报)
- **Watermark control**: Enable/disable watermarks (requires signed disclaimer for no-watermark)
- **Content safety**: Built-in content filtering for compliance

## Resource Links / 资源链接

| Resource        | Link                                                                      |
| --------------- | ------------------------------------------------------------------------- |
| **Get API Key** | [智谱开放平台 API Keys](https://bigmodel.cn/usercenter/proj-mgmt/apikeys) |
| **API Docs**    | [Image Generation / 图像生成](https://docs.bigmodel.cn/api-reference/%E6%A8%A1%E5%9E%8B-api/%E5%9B%BE%E5%83%8F%E7%94%9F%E6%88%90) |
| **Model Docs**  | [GLM-Image 模型文档](https://docs.bigmodel.cn/cn/guide/models/image-generation/glm-image) |

## Prerequisites / 前置条件

### API Key Setup / API Key 配置（Required / 必需）

脚本通过 `ZHIPU_API_KEY` 环境变量获取密钥，可与其他智谱技能复用同一个 key。
This script reads the key from the `ZHIPU_API_KEY` environment variable. Reusing the same key across Zhipu skills is optional.

**Get Key / 获取 Key：** Visit [智谱开放平台 API Keys](https://bigmodel.cn/usercenter/proj-mgmt/apikeys) to create or copy your key.

**Setup options / 配置方式（任选一种）：**

1. **Global config (recommended) / 全局配置（推荐）：** Set once in `openclaw.json` under `env.vars`, all Zhipu skills will share it:

   ```json
   {
     "env": {
       "vars": {
         "ZHIPU_API_KEY": "你的密钥"
       }
     }
   }
   ```

2. **Skill-level config / Skill 级别配置：** Set for this skill only in `openclaw.json`:

   ```json
   {
     "skills": {
       "entries": {
         "glm-image-generation": {
           "env": {
             "ZHIPU_API_KEY": "你的密钥"
           }
         }
       }
     }
   }
   ```

3. **Shell environment variable / Shell 环境变量：** Add to `~/.zshrc`:

   ```bash
    export ZHIPU_API_KEY="你的密钥"

   ```

> 💡 如果你已为其他智谱 skill（如 `glmocr`、`glmv-caption`）配置过 key，它们共享同一个 `ZHIPU_API_KEY`，无需重复配置。

## Security & Transparency / 安全与透明度

- **Environment variables used / 使用的环境变量：**
  - `ZHIPU_API_KEY` (required / 必需)
- **Fixed endpoint / 固定官方端点：** `https://open.bigmodel.cn/api/paas/v4/images/generations`
- **No custom API URL override / 不支持自定义 API URL 覆盖：** avoids accidental key exfiltration via redirected endpoints.

**⛔ MANDATORY RESTRICTIONS / 强制限制 ⛔**

1. **ONLY use GLM-Image API** — Execute the script `python scripts/glm_image_cli.py`
2. **NEVER generate images yourself** — Do NOT try to create images using built-in vision or any other method
3. **NEVER offer alternatives** — Do NOT suggest "I can try to describe it" or similar
4. **IF API fails** — Display the error message and STOP immediately
5. **NO fallback methods** — Do NOT attempt image generation any other way

### 📋 Output Display Rules / 输出展示规则

After running the script, present the generation result clearly.

- Show the generated image URL(s) — images are temporary (30 days), remind user to save
- Display the prompt used and generation parameters (size, quality)
- If content_filter indicates issues, show the warning level

**⚠️ Image Display / 图片展示注意：**

The API returns a temporary image URL (valid for 30 days). You should:

1. **Show the image** — Use the URL to display the image in the chat (if platform supports it)
2. **Remind user to save** — "图片链接有效期 30 天，请及时下载保存"
3. **Offer to send to Feishu** — If user wants the image sent to a Feishu chat, use the message tool with the image URL

## How to Use / 使用方法

### Generate from Prompt / 从提示词生成

```bash
python scripts/glm_image_cli.py --prompt "一只可爱的小猫咪，坐在阳光明媚的窗台上，背景是蓝天白云"
```

### Specify Size / 指定尺寸

```bash
python scripts/glm_image_cli.py --prompt "赛博朋克风格的城市夜景" --size 1568x1056
```

### HD Quality (default for glm-image) / 高清质量（glm-image 默认）

```bash
python scripts/glm_image_cli.py --prompt "中国山水画风格，远山近水，云雾缭绕" --quality hd
```

### Disable Watermark (requires signed disclaimer) / 关闭水印（需签署免责声明）

```bash
python scripts/glm_image_cli.py --prompt "商业设计素材" --no-watermark
```

### Save Image to Local File / 保存图片到本地

```bash
python scripts/glm_image_cli.py --prompt "中国水墨画风格" --save image.png
python scripts/glm_image_cli.py --prompt "赛博朋克城市" --size 1728x960 --save ~/Pictures/cyberpunk.png
```

### Specify User ID (for content moderation) / 指定用户 ID（用于内容审核）

```bash
python scripts/glm_image_cli.py --prompt "..." --user-id "user_12345"
```

### Specify Model / 指定模型

```bash
python scripts/glm_image_cli.py --prompt "..." --model glm-image
python scripts/glm_image_cli.py --prompt "..." --model cogview-4
```

## CLI Reference / CLI 参数

```
python {baseDir}/scripts/glm_image_cli.py --prompt TEXT [--model MODEL] [--size SIZE] [--quality QUALITY] [--no-watermark] [--user-id ID] [--save FILE]
```

| Parameter        | Required | Description                                                      |
| ---------------- | -------- | ---------------------------------------------------------------- |
| `--prompt`, `-p` | Yes      | Text description of the desired image / 图片的文本描述         |
| `--model`, `-m`  | No       | Model: glm-image (default), cogview-4, cogview-3-flash / 模型  |
| `--size`, `-s`   | No       | Image size (default: 1280x1280) / 图片尺寸                     |
| `--quality`, `-q`| No       | Quality: hd (default) or standard / 质量                        |
| `--no-watermark` | No       | Disable watermark (requires signed disclaimer) / 关闭水印       |
| `--user-id`      | No       | End-user ID for content moderation (6-128 chars) / 终端用户 ID  |
| `--save`         | No       | Save generated image to local file / 保存生成的图片到本地文件   |

## Supported Sizes / 支持的尺寸

**GLM-Image recommended sizes:**

| Size          | Aspect Ratio | Use Case                  |
| ------------- | ------------ | ------------------------- |
| 1280x1280     | 1:1          | Square (default)          |
| 1568×1056     | 3:2          | Landscape / 横向          |
| 1056×1568     | 2:3          | Portrait / 纵向           |
| 1472×1088     | ~4:3         | Wide landscape            |
| 1088×1472     | ~3:4         | Tall portrait             |
| 1728×960      | 16:9         | Ultra-wide landscape      |
| 960×1728      | 9:16         | Ultra-tall portrait       |

**Custom sizes / 自定义尺寸:**

- Width and height: 1024px - 2048px
- Both dimensions must be multiples of 32 / 长宽均需为 32 的整数倍
- Maximum total pixels: 2^22 (4,194,304 px) / 最大像素数不超过 2^22

## Response Format / 响应格式

**Official API Response:**

```json
{
  "created": 123,
  "data": [
    {
      "url": "<string>"
    }
  ],
  "content_filter": [
    {
      "role": "assistant",
      "level": 1
    }
  ]
}
```

**CLI Output Format:**

```json
{
  "ok": true,
  "model": "glm-image",
  "image_url": "https://open.bigmodel.cn/.../generated_image.png",
  "prompt": "一只可爱的小猫咪，坐在阳光明媚的窗台上，背景是蓝天白云",
  "size": "1280x1280",
  "quality": "hd",
  "created": 1710835200,
  "content_filter": [
    {
      "role": "assistant",
      "level": 3
    }
  ],
  "saved_file": "/Users/xxx/image.png",
  "error": null
}
```

Key fields:

- `ok` — whether generation succeeded
- `model` — model used for generation
- `image_url` — extracted from `data[0].url`, temporary URL (valid 30 days)
- `prompt` — the text prompt used
- `size` — generated image dimensions
- `quality` — hd or standard
- `created` — Unix timestamp when request was created
- `content_filter` — content safety analysis array (may be empty)
  - `role`: where the issue was detected (user/assistant/history)
  - `level`: severity 0-3 (0 = most severe, 3 = minor)
- `saved_file` — absolute path to saved local file (if `--save` was used)
- `error` — error details on failure

## Content Safety / 内容安全

The API includes content filtering. If issues are detected, `content_filter` will contain entries with:

- `role`: where the issue was detected (user/assistant/history)
- `level`: severity 0-3 (0 = most severe, 3 = minor)

**If level 0-1 detected:** Generation will fail, show error to user.
**If level 2-3 detected:** Generation may succeed, but warn user about potential issues.

## Error Handling / 错误处理

**API key not configured:**

```json
{
  "ok": false,
  "error": {
    "code": "MISSING_API_KEY",
    "message": "ZHIPU_API_KEY not configured. Get your API key at: https://bigmodel.cn/usercenter/proj-mgmt/apikeys"
  }
}
```

→ Show exact error to user, guide them to configure

**Authentication failed (401/403):**

```json
{
  "ok": false,
  "error": {
    "code": "authentication_error",
    "message": "令牌已过期或验证不正确",
    "status": 401
  }
}
```

→ API key invalid/expired → reconfigure

**Rate limit (429):**

```json
{
  "ok": false,
  "error": {
    "code": "rate_limit_exceeded",
    "message": "API rate limit exceeded. Please try again later.",
    "status": 429
  }
}
```

→ Quota exhausted → inform user to wait or check quota

**Content filter violation:**

```json
{
  "ok": false,
  "error": {
    "code": "content_filter_violation",
    "message": "Content safety check failed",
    "status": 400
  }
}
```

→ Explain that the prompt may contain inappropriate content

**Invalid size:**

```json
{
  "ok": false,
  "error": {
    "code": "INVALID_SIZE",
    "message": "Invalid size: 512x512 for model glm-image. Must be multiple of 32, 1024-2048px, max 2^22 pixels"
  }
}
```

→ Guide user to use valid size from the supported list

**Download failed:**

```json
{
  "ok": false,
  "error": {
    "code": "DOWNLOAD_FAILED",
    "message": "Failed to download image to image.png"
  }
}
```

→ Check file path permissions and disk space

**Network error:**

```json
{
  "ok": false,
  "error": {
    "code": "NETWORK_ERROR",
    "message": "Network error: [Errno 8] nodename nor servname provided, or not known"
  }
}
```

→ Check internet connection

## Prompt Tips / 提示词技巧

**Good prompts:**

- Specific details: "一只橘色的英国短毛猫，绿色眼睛，坐在木质窗台上"
- Style keywords: "赛博朋克风格", "中国水墨画", "油画质感", "3D 渲染"
- Lighting: "阳光明媚", "柔和的逆光", "电影感灯光"
- Composition: "特写镜头", "广角视角", "俯视角度"

**Avoid:**

- Vague descriptions: "好看的图片"
- Contradictory elements: "白天和夜晚同时"
- Too many subjects: Keep focus on 1-2 main elements
