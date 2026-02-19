output "content_understanding_endpoint" {
  description = "Endpoint URL for Azure AI Content Understanding"
  value       = azurerm_cognitive_account.content_understanding.endpoint
  sensitive   = false
}

output "content_understanding_key" {
  description = "Primary API key for Azure AI Content Understanding"
  value       = azurerm_cognitive_account.content_understanding.primary_access_key
  sensitive   = true
}

output "content_understanding_secondary_key" {
  description = "Secondary API key for Azure AI Content Understanding"
  value       = azurerm_cognitive_account.content_understanding.secondary_access_key
  sensitive   = true
}

output "resource_group_name" {
  description = "Name of the resource group"
  value       = azurerm_resource_group.content_understanding.name
}

output "resource_name" {
  description = "Name of the Azure AI Services resource"
  value       = azurerm_cognitive_account.content_understanding.name
}

output "location" {
  description = "Azure region where resources are deployed"
  value       = azurerm_resource_group.content_understanding.location
}

# Export commands for easy setup
output "export_commands" {
  description = "Commands to export environment variables"
  value       = <<-EOT
    # Copy and paste these commands to set up your environment:
    
    export AZURE_CONTENT_UNDERSTANDING_ENDPOINT="${azurerm_cognitive_account.content_understanding.endpoint}"
    export AZURE_CONTENT_UNDERSTANDING_KEY="${azurerm_cognitive_account.content_understanding.primary_access_key}"
    
    # Or add to your .env file:
    echo "AZURE_CONTENT_UNDERSTANDING_ENDPOINT=${azurerm_cognitive_account.content_understanding.endpoint}" >> .env
    echo "AZURE_CONTENT_UNDERSTANDING_KEY=${azurerm_cognitive_account.content_understanding.primary_access_key}" >> .env
  EOT
  sensitive   = true
}
