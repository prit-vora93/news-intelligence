import trafilatura

# A genuinely text-heavy article, not a short audio-brief page
url = "https://www.justsecurity.org/158424/early-edition-september-22-2026/"

print(f"Fetching: {url}")
downloaded = trafilatura.fetch_url(url)

if downloaded is None:
    print("Fetch failed -- trying a different URL might be needed.")
else:
    text = trafilatura.extract(downloaded)
    print()
    print("=" * 70)
    print("EXTRACTED TEXT (first 2000 characters)")
    print("=" * 70)
    print(text[:2000] if text else "No text extracted.")
    print()
    print(f"Total extracted length: {len(text) if text else 0} characters")
