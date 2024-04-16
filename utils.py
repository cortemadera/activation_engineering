import re

# Remove the [INST] section in Mistra response
def clean_text(text: str) -> str:
  pattern_1 = r"\[INST\].*\[/INST\]"
  text = re.sub(pattern_1, "", text, flags=re.DOTALL)
  pattern_2 = r"<s>(.*)</s>"
  text = re.sub(pattern_2, r'\1', text, flags=re.DOTALL)
  return text.strip()