Fixed inference.py JSON parser to use json.JSONDecoder.raw_decode() instead of rfind('}'), preventing "Extra data" errors when the model outputs multiple JSON objects in one response.
