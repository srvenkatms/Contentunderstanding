terraform {
  required_version = ">= 1.0"
  
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
}

provider "azurerm" {
  features {}
}

# Resource Group
resource "azurerm_resource_group" "content_understanding" {
  name     = var.resource_group_name
  location = var.location
  tags     = var.tags
}

# Azure AI Services (Cognitive Services) Multi-Service Account
# This resource supports Azure AI Content Understanding capabilities
resource "azurerm_cognitive_account" "content_understanding" {
  name                = var.resource_name
  location            = azurerm_resource_group.content_understanding.location
  resource_group_name = azurerm_resource_group.content_understanding.name
  kind                = "CognitiveServices"
  sku_name            = var.sku_name
  
  tags = var.tags
  
  # Enable public network access
  public_network_access_enabled = true
  
  # Identity configuration (optional, for managed identity scenarios)
  identity {
    type = "SystemAssigned"
  }
}
