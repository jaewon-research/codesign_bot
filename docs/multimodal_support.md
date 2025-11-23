# Multimodal Support in OASIS

OASIS now supports multimodal inputs, allowing you to provide images alongside text prompts to your social agents. This is particularly useful for simulations where visual context influences agent behavior.

## Overview

The multimodal support allows you to:
- Provide context images to agents at simulation start
- Use images from files, URLs, or base64-encoded strings
- Leverage Claude 3's vision capabilities on AWS Bedrock
- Mix multimodal and text-only agents in the same simulation

## Requirements

- **AWS Bedrock** with access to Claude 3 models (Sonnet, Haiku, or Opus)
- **boto3** package installed
- **Claude 3 models** - These are the only models in Bedrock that currently support vision

## Supported Image Formats

- JPEG/JPG
- PNG
- GIF
- WebP
- BMP

**Maximum image size:** 5MB (Bedrock API limitation)

## Quick Start

### 1. Basic Usage with File Path

```python
from oasis import UserInfo, SocialAgent, BedrockModelFactory
from oasis.social_platform import Channel

# Create Bedrock model (must be Claude 3)
model = BedrockModelFactory.create_claude_3_5_sonnet(
    region_name="us-east-2",
    temperature=0.7,
    max_tokens=4096
)

# Create user info with an image
user_info = UserInfo(
    user_name="ImageAgent",
    name="Image Agent",
    profile={
        "other_info": {
            "user_profile": "An agent that considers visual context",
            "gender": "Unknown",
            "age": "Unknown",
            "mbti": "Unknown",
            "country": "Unknown"
        }
    },
    recsys_type="reddit",
    system_image="./path/to/image.jpg",  # Path to your image
    image_type="file"  # 'file', 'url', or 'base64'
)

# Create agent - image will be included in system prompt
agent = SocialAgent(
    agent_id=0,
    user_info=user_info,
    channel=Channel(),
    model=model,
    available_actions=available_actions
)
```

### 2. Using Image URLs

```python
user_info = UserInfo(
    user_name="URLImageAgent",
    name="URL Image Agent",
    profile={...},
    recsys_type="reddit",
    system_image="https://example.com/image.jpg",
    image_type="url"
)
```

### 3. Using Base64-Encoded Images

```python
import base64

# Load and encode image
with open("image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

user_info = UserInfo(
    user_name="Base64Agent",
    name="Base64 Agent",
    profile={...},
    recsys_type="reddit",
    system_image=image_data,
    image_type="base64"
)
```

## Image Utilities

OASIS provides utility functions for working with images:

```python
from oasis.utils.image_utils import (
    detect_image_type,
    validate_image_file,
    encode_image_to_base64,
    create_image_content_block,
    prepare_multimodal_message
)

# Auto-detect image type
image_type = detect_image_type("./image.jpg")  # Returns 'file'

# Validate an image file
is_valid, error = validate_image_file("./image.jpg")
if not is_valid:
    print(f"Invalid image: {error}")

# Encode image to base64
base64_str = encode_image_to_base64("./image.jpg")

# Create image content block for API
image_block = create_image_content_block(
    image_source="./image.jpg",
    image_type="file"
)

# Prepare multimodal message
content = prepare_multimodal_message(
    text_content="Describe this image",
    image_source="./image.jpg"
)
```

## How It Works

### Under the Hood

1. **Image Loading**: When you create a `UserInfo` with a `system_image`, the image is processed based on its type:
   - **File**: Read from disk and base64-encoded
   - **URL**: Passed directly (for models that support URL images)
   - **Base64**: Used as-is

2. **System Message Preparation**: The `SocialAgent` detects the image and creates a multimodal system message:
   ```python
   [
       {
           "type": "image",
           "source": {
               "type": "base64",
               "media_type": "image/jpeg",
               "data": "<base64-encoded-data>"
           }
       },
       {
           "type": "text",
           "text": "Your system prompt text..."
       }
   ]
   ```

3. **Bedrock API Call**: The `BedrockModelBackend` handles multimodal content:
   - Extracts images from system messages
   - Prepends images to the first user message (Claude API requirement)
   - Formats according to Bedrock's Claude 3 API specification

### Claude on Bedrock Format

The correct format for images in Claude on Bedrock is:

```json
{
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 4096,
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": "<base64-encoded-image>"
                    }
                },
                {
                    "type": "text",
                    "text": "Your prompt text"
                }
            ]
        }
    ],
    "system": "Your system prompt (text only)"
}
```

**Important**: Claude on Bedrock only supports images in user/assistant messages, not in system messages. OASIS automatically handles this by moving system images to the first user message.

## Example: Complete Simulation

See `examples/reddit_simulation_multimodal.py` for a complete example.

```python
import asyncio
import oasis
from oasis import (
    ActionType, UserInfo, SocialAgent, 
    BedrockModelFactory, LLMAction
)
from oasis.social_platform import Channel

async def run_simulation():
    # Create model
    model = BedrockModelFactory.create_claude_3_5_sonnet(
        region_name="us-east-2"
    )
    
    # Create agent graph
    from oasis.social_agent import AgentGraph
    agent_graph = AgentGraph()
    channel = Channel()
    
    # Create agents with images
    for i, (name, image_path) in enumerate([
        ("Agent1", "./images/context1.jpg"),
        ("Agent2", "./images/context2.jpg"),
    ]):
        user_info = UserInfo(
            user_name=name,
            name=name,
            profile={"other_info": {...}},
            recsys_type="reddit",
            system_image=image_path,
            image_type="file"
        )
        
        agent = SocialAgent(
            agent_id=i,
            user_info=user_info,
            channel=channel,
            model=model,
            available_actions=ActionType.get_default_reddit_actions(),
            agent_graph=agent_graph
        )
        agent_graph.add_agent(agent)
    
    # Create environment
    env = oasis.make(
        agent_graph=agent_graph,
        platform=oasis.DefaultPlatformType.REDDIT,
        database_path="./simulation.db"
    )
    
    await env.reset()
    
    # Run simulation
    llm_actions = {agent: LLMAction() for _, agent in agent_graph.get_agents()}
    await env.step(llm_actions)
    
    await env.close()

asyncio.run(run_simulation())
```

## Best Practices

1. **Image Size**: Keep images under 5MB for best performance and API compliance
2. **Image Quality**: Use clear, high-quality images for better agent understanding
3. **Context Relevance**: Ensure images are relevant to the agent's role and simulation context
4. **Model Selection**: Only Claude 3 models support vision on Bedrock
5. **Error Handling**: Check image validity before running long simulations
6. **Cost Awareness**: Vision API calls are typically more expensive than text-only calls

## Troubleshooting

### Image Not Being Used

**Problem**: Agent seems to ignore the image context

**Solutions**:
- Verify the image path is correct
- Check that `image_type` is set correctly
- Ensure you're using a Claude 3 model (not Titan or others)
- Check logs for image processing errors

### Image Too Large Error

**Problem**: `Image too large` error

**Solutions**:
- Compress the image to under 5MB
- Use a lower resolution
- Convert to JPEG format for better compression

### Invalid Image Format

**Problem**: `Unsupported image format` error

**Solutions**:
- Convert to a supported format (JPEG, PNG, GIF, WebP, BMP)
- Check that the file is actually an image (correct extension)

### Model Doesn't Support Vision

**Problem**: Error about unsupported content type

**Solutions**:
- Ensure you're using Claude 3 Sonnet, Haiku, or Opus
- Check Bedrock model ID includes "claude-3"
- Verify your AWS region supports Claude 3

## API Reference

### UserInfo

```python
@dataclass
class UserInfo:
    user_name: str | None = None
    name: str | None = None
    description: str | None = None
    profile: dict[str, Any] | None = None
    recsys_type: str = "twitter"
    is_controllable: bool = False
    system_image: str | None = None  # Image path, URL, or base64 string
    image_type: str | None = None    # 'file', 'url', or 'base64'
```

### Image Utilities

```python
def detect_image_type(image_source: str) -> str:
    """Detect image type ('file', 'url', 'base64', 'unknown')"""

def validate_image_file(file_path: str) -> Tuple[bool, Optional[str]]:
    """Validate image file. Returns (is_valid, error_message)"""

def encode_image_to_base64(file_path: str) -> Optional[str]:
    """Encode image file to base64 string"""

def create_image_content_block(
    image_source: str, 
    image_type: Optional[str] = None
) -> Optional[dict]:
    """Create image content block for LLM API"""

def prepare_multimodal_message(
    text_content: str, 
    image_source: Optional[str] = None,
    image_type: Optional[str] = None
) -> list:
    """Prepare multimodal message with text and optional image"""
```

## Examples Directory

- `examples/reddit_simulation_multimodal.py` - Complete multimodal simulation example
- `examples/images/` - Place your test images here

## Further Reading

- [AWS Bedrock Claude Documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-claude.html)
- [Claude 3 Vision Capabilities](https://docs.anthropic.com/claude/docs/vision)
- [OASIS Core Documentation](./README.md)

