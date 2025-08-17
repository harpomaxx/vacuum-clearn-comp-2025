# Submission Instructions for Students

Welcome to the assignment repository! Please follow these instructions carefully to ensure your submission is accepted and graded properly.

---

## ✅ What to Submit

You must submit **exactly one Python file** that follows this naming format:

```
student_<name>_<lastname>_agent.py
```

### Examples of valid filenames:

- `student_john_doe_agent.py`
- `student_ana_garcia_agent.py`

### ❌ Invalid examples (will be rejected):

- `john_doe_agent.py` (missing 'student\_')- 
- `student_ohn_doe.py` (missing '\_agent')
- `student_john_agent.py` (missing lastname)
- `student_john_doe_extra_agent.py` (too many parts)
- `assignment.py` (wrong format)
- `caca.py` (unacceptable name)

---

## 🚫 What Not to Do

- Do **not** include multiple files
- Do **not** use spaces or special characters in filenames
- Do **not** use folders or subdirectories
- Do **not** rename your file after pushing

Any file that does not follow the required format will be **automatically deleted** by the system.

---

## 🛠️ How to Submit

1. Clone the repository:

   ```bash
   git clone <repo-url>
   cd <repo-folder>
   ```

2. Create your file with the correct format:

   ```bash
   touch student_john_doe.py
   ```

3. Stage, commit, and push:

   ```bash
   git add student_john_doe.py
   git commit -m "Add assignment file"
   git push
   ```

4. ✅ If your file is valid, it will stay in the repository. ❌ If your file is invalid, it will be automatically removed.

---

## ⚠️ Important Notes

- If your file is deleted by the system:

  - Correct the filename locally
  - Run: `git pull --rebase`
  - Re-add your corrected file
  - Push again

- The repository uses automated checks to enforce rules.

✨ Linux users: You can install a local pre-push Git hook to prevent pushing files with incorrect names. Run this command in your repo:

```bash
bash install-prepush-hook.sh
```
This will block any push that includes incorrectly named Python files.

Happy hacking!

