#!/usr/bin/env python3
"""
GLM-Image Generation CLI - ZhiPu GLM-Image API client

Usage:
    python glm_image_cli.py --prompt "A cute cat" [--size 1280x1280] [--quality hd] [--save image.png]

API Docs: https://docs.bigmodel.cn/api-reference/%E6%A8%A1%E5%9E%8B-api/%E5%9B%BE%E5%83%8F%E7%94%9F%E6%88%90
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate images from text prompts using GLM-Image API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python glm_image_cli.py --prompt "A cute cat sitting on windowsill"
  python glm_image_cli.py --prompt "Cyberpunk city" --size 1568x1056
  python glm_image_cli.py --prompt "Chinese landscape painting" --save image.png
        """,
    )

    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        required=True,
        help="Text description of the desired image (required)",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="glm-image",
        choices=["glm-image", "cogview-4-250304", "cogview-4", "cogview-3-flash"],
        help="Model to use (default: glm-image)",
    )
    parser.add_argument(
        "-s",
        "--size",
        type=str,
        default="1280x1280",
        help="Image size (default: 1280x1280)",
    )
    parser.add_argument(
        "-q",
        "--quality",
        type=str,
        default="hd",
        choices=["hd", "standard"],
        help="Image quality: hd or standard",
    )
    parser.add_argument(
        "--no-watermark",
        action="store_true",
        help="Disable watermark (requires signed disclaimer)",
    )
    parser.add_argument(
        "--user-id",
        type=str,
        default=None,
        help="End-user ID for content moderation (6-128 chars)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Save generated image to local file (e.g., image.png)",
    )

    return parser.parse_args()


def validate_size(size: str, model: str = "glm-image") -> bool:
    """Validate size format and dimensions.

    glm-image requirements:
    - Width and height: 1024px - 2048px
    - Both dimensions must be multiples of 32
    - Maximum total pixels: 2^22 (4,194,304)

    Other models (cogview-4, cogview-3-flash, etc.):
    - Width and height: 512px - 2048px
    - Both dimensions must be multiples of 16
    - Maximum total pixels: 2^21 (2,097,152)
    """
    try:
        parts = size.split("x")
        if len(parts) != 2:
            return False
        width = int(parts[0])
        height = int(parts[1])

        if model == "glm-image":
            # Must be multiple of 32
            if width % 32 != 0 or height % 32 != 0:
                return False
            # Must be between 1024-2048
            if not (1024 <= width <= 2048 and 1024 <= height <= 2048):
                return False
            # Max total pixels: 2^22 = 4194304
            if width * height > 4194304:
                return False
        else:
            # Other models: cogview-4, cogview-3-flash, etc.
            # Must be multiple of 16
            if width % 16 != 0 or height % 16 != 0:
                return False
            # Must be between 512-2048
            if not (512 <= width <= 2048 and 512 <= height <= 2048):
                return False
            # Max total pixels: 2^21 = 2097152
            if width * height > 2097152:
                return False

        return True
    except (ValueError, IndexError):
        return False


def validate_user_id(user_id: str) -> bool:
    """Validate user_id length (6-128 chars)."""
    if not user_id:
        return True
    return 6 <= len(user_id) <= 128


def download_image(url: str, save_path: str) -> bool:
    """Download image from URL to local file."""
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            with open(save_path, "wb") as f:
                f.write(response.read())
        return True
    except Exception:
        return False


def generate_image(
    api_key: str,
    model: str,
    prompt: str,
    size: str,
    quality: str,
    watermark: bool,
    user_id: str = None,
) -> dict:
    """Call GLM-Image API to generate image.

    Returns response in official format:
    {
        "created": 123,
        "data": [{"url": "..."}],
        "content_filter": [{"role": "assistant", "level": 1}]
    }
    """

    url = "https://open.bigmodel.cn/api/paas/v4/images/generations"

    payload = {
        "model": model,
        "prompt": prompt,
        "size": size,
        "quality": quality,
        "watermark_enabled": watermark,
    }

    if user_id:
        payload["user_id"] = user_id

    data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
            return {
                "ok": True,
                "result": result,
                "image_url": (
                    result.get("data", [{}])[0].get("url")
                    if result.get("data")
                    else None
                ),
                "prompt": prompt,
                "size": size,
                "quality": quality,
                "created": result.get("created"),
                "content_filter": result.get("content_filter", []),
            }
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8") if e.fp else ""
        error_detail = ""
        error_code = str(e.code)

        try:
            error_json = json.loads(error_body)
            if isinstance(error_json, dict):
                error_detail = str(
                    error_json.get("error", {}).get("message", "")
                ).strip()
                error_code = str(error_json.get("error", {}).get("code", str(e.code)))
        except Exception:
            pass

        if not error_detail:
            error_detail = (error_body[:200] or "No response body").strip()

        return {
            "ok": False,
            "error": {"code": error_code, "message": error_detail, "status": e.code},
        }
    except urllib.error.URLError as e:
        return {
            "ok": False,
            "error": {
                "code": "NETWORK_ERROR",
                "message": f"Network error: {str(e.reason)}",
            },
        }
    except Exception as e:
        return {"ok": False, "error": {"code": "UNKNOWN_ERROR", "message": str(e)}}


def main():
    args = parse_args()

    # Get API key
    api_key = os.environ.get("ZHIPU_API_KEY")
    if not api_key:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": {
                        "code": "MISSING_API_KEY",
                        "message": "ZHIPU_API_KEY not configured. Get your API key at: https://bigmodel.cn/usercenter/proj-mgmt/apikeys",
                    },
                },
                ensure_ascii=False,
            )
        )
        sys.exit(1)

    # Validate size based on model
    if not validate_size(args.size, args.model):
        if args.model == "glm-image":
            size_msg = "Must be multiple of 32, 1024-2048px, max 2^22 pixels"
        else:
            size_msg = "Must be multiple of 16, 512-2048px, max 2^21 pixels"
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": {
                        "code": "INVALID_SIZE",
                        "message": f"Invalid size: {args.size} for model {args.model}. {size_msg}",
                    },
                },
                ensure_ascii=False,
            )
        )
        sys.exit(1)

    # Validate user_id
    if args.user_id and not validate_user_id(args.user_id):
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": {
                        "code": "INVALID_USER_ID",
                        "message": f"Invalid user_id: {args.user_id}. Must be 6-128 characters.",
                    },
                },
                ensure_ascii=False,
            )
        )
        sys.exit(1)

    # Generate image
    result = generate_image(
        api_key=api_key,
        model=args.model,
        prompt=args.prompt,
        size=args.size,
        quality=args.quality,
        watermark=not args.no_watermark,
        user_id=args.user_id,
    )

    # Download image if --save specified
    saved_file = None
    if result["ok"] and args.save and result.get("image_url"):
        if download_image(result["image_url"], args.save):
            saved_file = os.path.abspath(args.save)
        else:
            result["ok"] = False
            result["error"] = {
                "code": "DOWNLOAD_FAILED",
                "message": f"Failed to download image to {args.save}",
            }

    # Output result (matching official API format where applicable)
    output_data = {
        "ok": result["ok"],
        "model": args.model,
        "image_url": result.get("image_url"),
        "prompt": result.get("prompt"),
        "size": result.get("size"),
        "quality": result.get("quality"),
        "created": result.get("created"),
        "content_filter": result.get("content_filter", []),
        "saved_file": saved_file,
        "error": result.get("error"),
    }

    # Print to stdout (compact JSON)
    print(json.dumps(output_data, ensure_ascii=False))

    # Exit with error code if failed
    if not result["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
