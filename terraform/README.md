# Terraform Configuration for Azure AI Content Understanding

This directory contains Terraform configuration to provision Azure resources required for Azure AI Content Understanding.

## Prerequisites

1. **Azure CLI** - [Install Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli)
2. **Terraform** - [Install Terraform](https://www.terraform.io/downloads) (version >= 1.0)
3. **Azure Subscription** - Active Azure subscription with permissions to create resources

## Resources Created

- **Azure Resource Group** - Container for all resources
- **Azure AI Services (Cognitive Services)** - Multi-service account that supports Azure AI Content Understanding

## Quick Start

### 1. Authenticate with Azure

```bash
az login
az account set --subscription "<your-subscription-id>"
```

### 2. Initialize Terraform

```bash
cd terraform
terraform init
```

### 3. Review and Customize Variables (Optional)

Edit `terraform.tfvars` or pass variables via command line:

```bash
# Create terraform.tfvars file
cat > terraform.tfvars <<EOF
resource_group_name = "rg-my-content-understanding"
location           = "eastus"
resource_name      = "my-content-understanding"
sku_name          = "S0"
tags = {
  Environment = "Production"
  Project     = "MyProject"
}
EOF
```

### 4. Plan Deployment

```bash
terraform plan
```

### 5. Deploy Resources

```bash
terraform apply
```

Type `yes` when prompted to confirm.

### 6. Get Outputs

After deployment completes, retrieve the endpoint and key:

```bash
# View all outputs
terraform output

# View specific output
terraform output content_understanding_endpoint

# View sensitive outputs (API keys)
terraform output -raw content_understanding_key

# Get export commands
terraform output -raw export_commands
```

### 7. Set Up Environment Variables

Copy the export commands from the output:

```bash
# Export to current shell
export AZURE_CONTENT_UNDERSTANDING_ENDPOINT="<your-endpoint>"
export AZURE_CONTENT_UNDERSTANDING_KEY="<your-key>"

# Or add to .env file
echo "AZURE_CONTENT_UNDERSTANDING_ENDPOINT=<your-endpoint>" >> ../.env
echo "AZURE_CONTENT_UNDERSTANDING_KEY=<your-key>" >> ../.env
```

Or use the automated command:

```bash
terraform output -raw export_commands | bash
```

## Configuration Options

### Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `resource_group_name` | Name of the resource group | `rg-content-understanding` | No |
| `location` | Azure region (must support Content Understanding) | `eastus` | No |
| `resource_name` | Name of the AI Services resource | `ai-content-understanding` | No |
| `sku_name` | Pricing tier (F0=Free, S0=Standard) | `S0` | No |
| `tags` | Resource tags | See variables.tf | No |

### Supported Regions

Azure AI Content Understanding is available in specific regions. Recommended regions:
- `eastus`
- `westus2`
- `westeurope`
- `northeurope`

Check [Azure documentation](https://learn.microsoft.com/azure/ai-services/document-intelligence/overview) for the latest region availability.

### SKU Options

- **F0 (Free)** - Limited transactions, good for development/testing
- **S0 (Standard)** - Pay-as-you-go, suitable for production

## Outputs

| Output | Description | Sensitive |
|--------|-------------|-----------|
| `content_understanding_endpoint` | API endpoint URL | No |
| `content_understanding_key` | Primary API key | Yes |
| `content_understanding_secondary_key` | Secondary API key | Yes |
| `resource_group_name` | Resource group name | No |
| `resource_name` | Resource name | No |
| `location` | Deployment region | No |
| `export_commands` | Shell commands to set environment variables | Yes |

## Managing the Deployment

### View State

```bash
terraform show
```

### Update Resources

1. Modify variables or configuration
2. Run `terraform plan` to preview changes
3. Run `terraform apply` to apply changes

### Destroy Resources

**Warning**: This will delete all resources and cannot be undone.

```bash
terraform destroy
```

## Troubleshooting

### Resource Name Already Exists

If you get an error about resource names already existing, either:
1. Choose a different `resource_name`
2. Use an existing resource by importing it

### Region Not Supported

If you get errors about features not being available:
1. Check that your region supports Azure AI Content Understanding
2. Try a different region (e.g., `eastus`, `westus2`)

### Authentication Errors

Ensure you're logged in and have selected the correct subscription:

```bash
az account show
az account list --output table
az account set --subscription "<subscription-id>"
```

## Azure AI Foundry / Azure OpenAI (Optional)

Azure AI Content Understanding can optionally use Azure OpenAI models for enhanced semantic analysis and embeddings. This is **not required** for basic clause checking functionality but can improve accuracy.

### When to Add Azure OpenAI

Consider adding Azure OpenAI if you need:
- Enhanced semantic similarity detection
- Advanced embedding models (e.g., text-embedding-3-large)
- LLM-based completion models (e.g., GPT-4)

### How to Provision Azure OpenAI

If you decide to add Azure OpenAI support, add this to your `main.tf`:

```hcl
# Azure OpenAI Service (Optional - for enhanced semantic analysis)
resource "azurerm_cognitive_account" "openai" {
  name                = "${var.resource_name}-openai"
  location            = azurerm_resource_group.content_understanding.location
  resource_group_name = azurerm_resource_group.content_understanding.name
  kind                = "OpenAI"
  sku_name            = "S0"
  
  tags = var.tags
}

# Deploy embedding model
resource "azurerm_cognitive_deployment" "embedding" {
  name                 = "text-embedding-3-small"
  cognitive_account_id = azurerm_cognitive_account.openai.id
  
  model {
    format  = "OpenAI"
    name    = "text-embedding-3-small"
    version = "1"
  }
  
  sku {
    name = "Standard"
    capacity = 1
  }
}

output "openai_endpoint" {
  value     = azurerm_cognitive_account.openai.endpoint
  sensitive = false
}

output "openai_key" {
  value     = azurerm_cognitive_account.openai.primary_access_key
  sensitive = true
}
```

Then set additional environment variables:

```bash
export AZURE_OPENAI_ENDPOINT="<openai-endpoint>"
export AZURE_OPENAI_KEY="<openai-key>"
```

## Next Steps

After deploying the infrastructure:

1. **Set up environment variables** (see step 7 above)
2. **Update analyzer configuration** using `update_analyzer.py`
3. **Run clause checks** using `check_clause_rest.py`

See the main [README.md](../README.md) for usage instructions.

## Additional Resources

- [Azure AI Services Documentation](https://learn.microsoft.com/azure/ai-services/)
- [Azure AI Document Intelligence](https://learn.microsoft.com/azure/ai-services/document-intelligence/)
- [Terraform Azure Provider](https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs)
