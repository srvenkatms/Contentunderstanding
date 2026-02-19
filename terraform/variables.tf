variable "resource_group_name" {
  description = "Name of the Azure Resource Group"
  type        = string
  default     = "rg-content-understanding"
}

variable "location" {
  description = "Azure region for resources. Use regions that support Azure AI Content Understanding (e.g., eastus, westus2, westeurope)"
  type        = string
  default     = "eastus"
}

variable "resource_name" {
  description = "Name of the Azure AI Services resource"
  type        = string
  default     = "ai-content-understanding"
  
  validation {
    condition     = length(var.resource_name) >= 2 && length(var.resource_name) <= 64
    error_message = "Resource name must be between 2 and 64 characters."
  }
}

variable "sku_name" {
  description = "SKU name for the Azure AI Services resource. Options: F0 (Free), S0 (Standard)"
  type        = string
  default     = "S0"
  
  validation {
    condition     = contains(["F0", "S0"], var.sku_name)
    error_message = "SKU name must be either F0 (Free) or S0 (Standard)."
  }
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default = {
    Environment = "Development"
    Project     = "ContentUnderstanding"
    ManagedBy   = "Terraform"
  }
}
