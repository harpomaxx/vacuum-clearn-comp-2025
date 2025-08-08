#!/bin/bash

HOOK_PATH=".git/hooks/pre-push"

echo "🛠️ Installing pre-push hook..."

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

# Get all staged .py files
STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACM | grep '\.py$')

for file in $STAGED_FILES; do
  filename=$(basename "$file")

  if [[ ! "$filename" =~ $VALID_REGEX ]]; then
    echo "❌ Invalid filename: $file"
    echo "   🧪 Expected format: student_name_lastname.py"
    EXIT=1
  else
    echo "✅ Valid: $file"
  fi
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

