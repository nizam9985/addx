require('dotenv').config();
const { DefaultAzureCredential } = require('@azure/identity');
const { SecretClient } = require('@azure/keyvault-secrets');

async function main() {
  // 1. Load the Key Vault URL from the .env file
  const keyVaultUrl = process.env.KEY_VAULT_URL;
  if (!keyVaultUrl) {
    throw new Error("KEY_VAULT_URL is empty. Please set it in your .env file.");
  }

  console.log(`Connecting to Key Vault: ${keyVaultUrl}`);

  // 2. Authenticate to Azure
  // DefaultAzureCredential is highly recommended. It attempts to authenticate via:
  // - Environment variables (AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID)
  // - Managed Identity (if running on Azure)
  // - Azure CLI (if you ran `az login` locally)
  const credential = new DefaultAzureCredential();

  // 3. Create the Secret Client
  const client = new SecretClient(keyVaultUrl, credential);

  // --- Sample Operations ---

  const secretName = "MySampleSecret";
  const secretValue = "HelloAzureKeyVault123!";

  try {
    // 4. Create (Set) a Secret
    console.log(`Setting secret '${secretName}'...`);
    const setSecretResult = await client.setSecret(secretName, secretValue);
    console.log(`Successfully set secret! (Version: ${setSecretResult.properties.version})`);

    // 5. Read (Get) a Secret
    console.log(`Retrieving secret '${secretName}'...`);
    const retrievedSecret = await client.getSecret(secretName);
    console.log(`The value of the secret is: ${retrievedSecret.value}`);

    // 6. Delete a Secret (Optional)
    // Note: Key Vaults usually have "soft-delete" enabled by default, 
    // so deleting a secret moves it to a recoverable state before permanent deletion.
    console.log(`Deleting secret '${secretName}'...`);
    const deletePoller = await client.beginDeleteSecret(secretName);
    await deletePoller.pollUntilDone();
    console.log("Secret deleted successfully.");

  } catch (error) {
    console.error("Error interacting with Key Vault:", error.message);
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
