import os
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

def main():
    # 1. Load environment variables from the .env file
    load_dotenv()
    
    # 2. Get the Key Vault URL
    key_vault_url = os.getenv("KEY_VAULT_URL")
    if not key_vault_url:
        raise ValueError("KEY_VAULT_URL is empty. Please set it in your .env file.")
        
    print(f"Connecting to Key Vault: {key_vault_url}")

    # 3. Authenticate to Azure
    # DefaultAzureCredential tries multiple authentication methods including:
    # - Environment variables (AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID)
    # - Azure CLI credentials (if you ran `az login` locally)
    # - Managed Identity (if running on Azure)
    credential = DefaultAzureCredential()

    # 4. Create the Secret Client
    client = SecretClient(vault_url=key_vault_url, credential=credential)

    # --- Sample Operations ---
    secret_name = "MySampleSecret"
    secret_value = "HelloAzureKeyVaultPython123!"

    try:
        # 5. Create (Set) a Secret
        print(f"Setting secret '{secret_name}'...")
        set_secret_result = client.set_secret(secret_name, secret_value)
        print(f"Successfully set secret! (Version: {set_secret_result.version})")

        # 6. Read (Get) a Secret
        print(f"Retrieving secret '{secret_name}'...")
        retrieved_secret = client.get_secret(secret_name)
        print(f"The value of the secret is: {retrieved_secret.value}")

        # 7. Delete a Secret (Optional)
        # Note: Key Vaults usually have "soft-delete" enabled by default, 
        # so deleting a secret moves it to a recoverable state before permanent deletion.
        print(f"Deleting secret '{secret_name}'...")
        delete_poller = client.begin_delete_secret(secret_name)
        delete_poller.wait()
        print("Secret deleted successfully.")

    except Exception as e:
        print(f"Error interacting with Key Vault: {e}")

if __name__ == "__main__":
    main()
