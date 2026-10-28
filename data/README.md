# Local data setup

Course-supplied assets are deliberately ignored by Git. Verify the course's
redistribution permission before publishing them. Renaming a file does not
change its license. The code can use equivalent data that you own instead.

The existing workspace already uses this layout:

```text
data/
  raw/
    restaurant_guide.txt
    recipes.json
    user_reviews.json
    recipe_images/recipe1.png ... recipe109.png
  processed/
    restaurants.json
    restaurant.schema.json
    recipes_enriched.json
    reviews_enriched.json
    review_images/
    backups/
```

For another checkout, copy your authorized local files into these paths.
Original-to-current mapping:

| Original | Current |
|---|---|
| California-Culinary-Map.txt | raw/restaurant_guide.txt |
| Recipes.json | raw/recipes.json |
| Synthetic-User-Reviews.json | raw/user_reviews.json |
| synthetic_recipe_images/ | raw/recipe_images/ |
| structured_restaurant_data.json | processed/restaurants.json |
| structured_restaurant_data.schema.json | processed/restaurant.schema.json |
| augmented_food_recipe.json | processed/recipes_enriched.json |
| augmented_user_review.json | processed/reviews_enriched.json |

## Record format

All data JSON files are arrays of objects, except the schema file.

- Restaurants: integer `itemId`, `name`, `location`, `type`, `food_style`,
  numeric `rating` (0-5 or null), integer `price_range` (1-4 or null),
  `signatures` (list), `vibe`, `environment`, `shortcomings` (list).
- Recipes: integer `id`, `name`, `cuisine`, `ingredients` (list of strings),
  `directions`, `total_time`, plus `image_description` in the enriched output.
  The corresponding PNG is named `recipe<id>.png`.
- Reviews: `userId`, `itemId` matching a restaurant, `title`, `text`, `rating`,
  optional `date`, image paths as used by the caption script, and
  `image_captions` in the enriched output.

After restoring the processed files and images, run `make index`.
The old `chroma_multimodal` store is not used: indexes are rebuilt locally
under `vector_store` with portable image paths.
