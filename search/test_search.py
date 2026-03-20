from search.search_query import search_claims

results = search_claims("Allina")

print("Search Results:")
for r in results:
    print("-" * 40)
    print(r["source_file"])
    print(r["content_preview"])
