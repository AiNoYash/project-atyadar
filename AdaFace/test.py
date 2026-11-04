from AdaFaceMatcher import AdaFaceMatcher

# 1. Initialize once when the server starts
# ? check readme.md for weight_path
matcher = AdaFaceMatcher(weight_path='adaface_ir50_ms1mv2.ckpt', database_path='database_embeddings.pt')

# If they ever need to rebuild the database from scratch:
# AdaFaceMatcher.create_database_embeddings('indian_faces_50', matcher.model)

matcher.guide()

best_match, highest_sim = matcher.find_best_match("src/image_blurriest.jpg")
print(best_match, highest_sim)

best_match, highest_sim = matcher.find_best_match("src/image_blurry.jpg")
print(best_match, highest_sim)


# ? The results should improve
print(matcher.giveGuidance())