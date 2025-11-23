# AWS Bedrock Integration with OASIS

This document explains how to use AWS Bedrock with OASIS for social media simulations.

## Overview

OASIS now supports AWS Bedrock for LLM inference, allowing you to use models like Claude, Amazon Titan, and Jurassic-2 for social media simulations.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements_bedrock.txt
```

### 2. Configure AWS Credentials

Set up your AWS credentials using one of these methods:

**Option A: Environment Variables**
```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
```

**Option B: AWS CLI**
```bash
aws configure
```

**Option C: IAM Roles (for EC2/ECS)**
```bash
# No additional setup needed if using IAM roles
```

### 3. Enable Bedrock Access

1. Go to AWS Bedrock console
2. Enable the models you want to use
3. Request access to specific models if needed

## Usage

### Basic Usage

```python
import oasis
from oasis import BedrockModelFactory, ActionType, generate_reddit_agent_graph

# Create a Bedrock model
bedrock_model = BedrockModelFactory.create_claude_3_sonnet(
    region_name="us-east-1",
    temperature=0.7,
    max_tokens=4096
)

# Create agent graph
agent_graph = await generate_reddit_agent_graph(
    profile_path="./data/reddit/user_data_36.json",
    model=bedrock_model,
    available_actions=ActionType.get_default_reddit_actions(),
)

# Run simulation
env = oasis.make(
    agent_graph=agent_graph,
    platform=oasis.DefaultPlatformType.REDDIT,
    database_path="./simulation.db",
)
```

### Available Models

#### Claude Models
```python
# Claude 3.5 Sonnet (recommended)
claude_sonnet = BedrockModelFactory.create_claude_3_5_sonnet(
    region_name="us-east-1",
    temperature=0.7
)

# Claude 3 Haiku (faster, cheaper)
claude_haiku = BedrockModelFactory.create_claude_3_haiku(
    region_name="us-east-1",
    temperature=0.7
)
```

#### Amazon Titan Models
```python
# Titan Text Large
titan_large = BedrockModelFactory.create_titan_text_large(
    region_name="us-east-1",
    temperature=0.7
)
```

#### AI21 Jurassic Models
```python
# Jurassic-2 Ultra
j2_ultra = BedrockModelFactory.create_j2_ultra(
    region_name="us-east-1",
    temperature=0.7
)
```

### Load Balancing with Multiple Models

```python
from camel.models import ModelManager

# Create multiple models
claude_sonnet = BedrockModelFactory.create_claude_3_5_sonnet()
claude_haiku = BedrockModelFactory.create_claude_3_haiku()

# Create model manager
model_manager = ModelManager(
    models=[claude_sonnet, claude_haiku],
    scheduling_strategy='round_robin'
)

# Use with OASIS
agent_graph = await generate_reddit_agent_graph(
    profile_path="./data/reddit/user_data_36.json",
    model=model_manager,
    available_actions=ActionType.get_default_reddit_actions(),
)
```

## Configuration Options

### Model Parameters

```python
bedrock_model = BedrockModelFactory.create(
    model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
    region_name="us-east-1",
    temperature=0.7,        # 0.0 to 1.0
    max_tokens=4096,       # Maximum tokens to generate
)
```

### Regional Availability

Different models are available in different regions:

- **us-east-1**: All models available
- **us-west-2**: Most models available
- **eu-west-1**: Limited model availability
- **ap-southeast-1**: Limited model availability

## Cost Considerations

### Model Pricing (as of 2024)

| Model | Input (per 1K tokens) | Output (per 1K tokens) |
|-------|----------------------|------------------------|
| Claude 3.5 Sonnet | $0.003 | $0.015 |
| Claude 3 Haiku | $0.00025 | $0.00125 |
| Titan Text Large | $0.0008 | $0.0016 |
| Jurassic-2 Ultra | $0.0125 | $0.0125 |

### Cost Optimization Tips

1. **Use Haiku for simple tasks**: Claude 3 Haiku is much cheaper
2. **Limit max_tokens**: Set appropriate limits to control costs
3. **Use load balancing**: Distribute requests across models
4. **Monitor usage**: Set up CloudWatch alarms for cost monitoring

## Error Handling

```python
try:
    bedrock_model = BedrockModelFactory.create_claude_3_sonnet()
    # Use model...
except Exception as e:
    print(f"Bedrock error: {e}")
    # Handle error...
```

### Common Errors

1. **AccessDeniedException**: Model not enabled in your account
2. **ValidationException**: Invalid model parameters
3. **ThrottlingException**: Rate limit exceeded
4. **ServiceQuotaExceededException**: Quota exceeded

## Performance Tips

1. **Use appropriate regions**: Choose regions close to your infrastructure
2. **Batch requests**: Use ModelManager for load balancing
3. **Monitor latency**: Use CloudWatch to monitor performance
4. **Cache responses**: Implement caching for repeated requests

## Security Considerations

1. **IAM Permissions**: Use least privilege principle
2. **VPC Endpoints**: Use VPC endpoints for private access
3. **Encryption**: Enable encryption in transit and at rest
4. **Audit Logging**: Enable CloudTrail for audit logs

## Troubleshooting

### Common Issues

1. **"No credentials found"**
   - Check AWS credentials configuration
   - Verify IAM permissions

2. **"Model not found"**
   - Check model ID spelling
   - Verify model is enabled in your account

3. **"Rate limit exceeded"**
   - Implement exponential backoff
   - Use multiple models for load balancing

4. **"Region not supported"**
   - Check model availability in your region
   - Use us-east-1 for maximum compatibility

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# This will show detailed Bedrock API calls
bedrock_model = BedrockModelFactory.create_claude_3_sonnet()
```

## Examples

See `examples/reddit_simulation_bedrock.py` for a complete example.

## Support

For issues related to:
- **OASIS**: Check the main OASIS documentation
- **AWS Bedrock**: Check AWS Bedrock documentation
- **boto3**: Check boto3 documentation
