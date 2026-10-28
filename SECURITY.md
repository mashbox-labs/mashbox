# Security

## Report a vulnerability

Use GitHub's private reporting: open the repo's **Security** tab and choose **Report a vulnerability**. Or email hello@mashbox.gg with "SECURITY" in the subject. Please don't open a public issue.

Include the version, the steps to reproduce it and what an attacker gets.

## In scope

- the `mashbox` package and CLI, the demo apps and the workflows in this repo
- the compiled kernel bundles in our Hugging Face repos: crashes, memory safety, anything that runs code it shouldn't
- files we publish on Hugging Face, for example a file that runs code when loaded

## Not in scope

Google Cloud, JAX, libtpu and llama.cpp have their own maintainers; report their bugs to them.

## Supported versions

Fixes go into the latest release.
