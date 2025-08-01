from cryptography.fernet import Fernet

def generate_key():
    """Generate a symmetric encryption key."""
    return Fernet.generate_key()

def encrypt_file(input_file, output_file, key):
    """Encrypt the input file and write to the output file."""
    fernet = Fernet(key)
    with open(input_file, 'rb') as file:
        original = file.read()
    encrypted = fernet.encrypt(original)
    with open(output_file, 'wb') as file:
        file.write(encrypted)

def decrypt_file(encrypted_file, output_file, key):
    """Decrypt the encrypted file and write to the output file."""
    fernet = Fernet(key)
    with open(encrypted_file, 'rb') as file:
        encrypted = file.read()
    decrypted = fernet.decrypt(encrypted)
    with open(output_file, 'wb') as file:
        file.write(decrypted)

if __name__ == "__main__":
    # Example usage
    input_file = 'users_comparison.json'
    encrypted_file = 'example.encrypted.txt'
    decrypted_file = 'example.decrypted.json'

    key = generate_key()
    print(f"Encryption Key (save this!): {key.decode()}")

    encrypt_file(input_file, encrypted_file, key)
    print(f"Encrypted '{input_file}' to '{encrypted_file}'")

    decrypt_file(encrypted_file, decrypted_file, key)
    print(f"Decrypted '{encrypted_file}' to '{decrypted_file}'")
