# Validating citations after the fact

The answer is parsed for citation markers and each is checked against the length
of the sources list. Out-of-range markers are invalid citations.

This is a separate pass from generation on purpose. Asking the model to
self-report whether it cited correctly measures the model's confidence, not its
citations.

Three states are distinguished and never collapsed: cited and valid, refused,
and unsupported.
