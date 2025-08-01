import json
import random
import string
from eth_hash.auto import keccak

def generate_password(length=10):
    chars = string.ascii_letters + string.digits
    return ''.join(random.choice(chars) for _ in range(length))

def keccak_hash(data: str) -> str:
    return keccak(data.encode()).hex()

def build_merkle_tree(leaves):
    tree = [leaves]
    while len(leaves) > 1:
        new_level = []
        for i in range(0, len(leaves), 2):
            left = leaves[i]
            right = leaves[i + 1] if i + 1 < len(leaves) else left
            combined = keccak(bytes.fromhex(left + right)).hex()
            new_level.append(combined)
        tree.append(new_level)
        leaves = new_level
    return tree

with open("data.json", "r") as file:
    raw_data = json.load(file)

users = {}
credentials_list = []

for user in raw_data:
    user_id = user.get("User ID")
    password = generate_password()
    users[user_id] = password
    credentials_list.append({
        "User ID": user_id,
        "Password": password
    })

with open("user_credentials.json", "w") as cred_file:
    json.dump(credentials_list, cred_file, indent=4)

hashed_leaves = [keccak_hash(uid + pwd) for uid, pwd in users.items()]

merkle_tree = build_merkle_tree(hashed_leaves)
merkle_root = merkle_tree[-1][0] if merkle_tree else None

output = {
    "users": users,
    "hashed_leaves": hashed_leaves,
    "merkle_tree": merkle_tree,
    "merkle_root": merkle_root
}

with open("merkle_tree_keccak.json", "w") as f:
    json.dump(output, f, indent=4)

print("✅ Merkle root:", merkle_root)
print("📁 Merkle tree saved to 'merkle_tree_keccak.json'")
print("🔐 User credentials saved to 'user_credentials.json'")
