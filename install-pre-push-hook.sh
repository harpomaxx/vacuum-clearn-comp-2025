#!/bin/bash

HOOK_PATH=".git/hooks/pre-push"

echo "🛠️  Installing pre-push hook..."

# Ensure this is a Git repo
if [ ! -d ".git" ]; then
  echo "❌ Not a Git repository."
  exit 1
fi

# Create hooks dir if not exists
mkdir -p .git/hooks

# Write the hook file
cat > "$HOOK_PATH" << 'EOF'
#!/bin/bash

echo "🔍 Validating Python filenames before push..."

VALID_REGEX="^student_[a-zA-Z]+_[a-zA-Z]+\.py$"
EXIT=0

while read -r local_ref local_sha remote_ref remote_sha; do
  # When pushing a new branch, remote_sha is all 0s
  if [[ "$remote_sha" =~ ^0+$ ]]; then
    commit_range="$local_sha"
  else
    commit_range="$remote_sha..$local_sha"
  fi

  # Get list of added or modified .py files in the commit range
  FILES=$(git diff --name-only --diff-filter=AM "$commit_range" | grep '\.py$')

  for file in $FILES; do
    filename=$(basename "$file")

    if [[ ! "$filename" =~ $VALID_REGEX ]]; then
      echo "❌ Invalid filename: $file"
      echo "   🧪 Expected format: student_name_lastname.py"
      EXIT=1
    else
      echo "✅ Valid: $file"
    fi
  done
done

if [[ $EXIT -ne 0 ]]; then
  echo "🚫 Push blocked due to invalid filenames."
  exit 1
fi

echo "✅ All filenames valid. Proceeding with push."
exit 0
EOF

# Make it executable
chmod +x "$HOOK_PATH"

echo "✅ Pre-push hook installed successfully at $HOOK_PATH"



