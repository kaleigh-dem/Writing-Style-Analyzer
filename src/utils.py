import openai
import json
from config import OPENAI_API_KEY, MAX_INPUT_TOKENS, MAX_GENERATED_TOKENS

# Initialize OpenAI client lazily (only when function is called)
def generate_weighted_text(prompt, author_probabilities, topic, use_personal_style):
    """Generates text with or without considering personal writing style."""

    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

    # 🔹 Filter top 3 dominant authors (only those with >10% influence)
    dominant_authors = {author: weight for author, weight in author_probabilities.items() if weight >= 10}
    dominant_authors = dict(sorted(dominant_authors.items(), key=lambda item: item[1], reverse=True)[:3])

    # If no authors qualify, use the highest-weighted author
    if not dominant_authors:
        dominant_authors = {max(author_probabilities, key=author_probabilities.get): 100.0}

    # 🔹 Format style weights
    style_weights = ", ".join([f"{author} ({weight:.2f}%)" for author, weight in dominant_authors.items()])

    # 🔹 System prompts (kept **identical** to your original code)
    if use_personal_style:
        system_prompt = (
                "You are a creative writing AI with expertise in literary styles."
                "The user wants a passage that blends the user’s **personal writing style** "
                f"with these dominant author influences (listed with their respective percentages): {style_weights}."
                "Please follow these steps:"
                "1. **Identify less than 10 defining features** of the blended style. Provide these as a concise, comma-separated list. "
                "   - The percentages shown must exactly match the style weights given (e.g., Doyle (98.36%))."
                "   - At least half of these defining features should come from the user’s personal style. "
                "     Indicate personal style elements with '(Your Style)' and author elements with '(AuthorName weight%)'."
                "2. **Generate a ~200-word passage** that incorporates:"
                "   - The user’s personal style traits (you must use them prominently)."
                "   - The dominant author traits in proportion to the weights (higher-weighted authors influence the text more)."
                "   - Natural continuity from the user’s provided passage (if any text was given)."
                "3. **Explain in 3–4 sentences** how the generated text reflects both the user's style and the author influences. "
                "Keep this explanation concise and focused on specific style features."
                "4. **Output everything as a valid JSON object** with the following structure:"
                "{"
                '  "style_elements": "comma-separated stylistic traits here",'
                '  "generated_passage": "Generated text here",'
                '  "style_explanation": "Brief explanation of stylistic blend here"'
                "}"
                "Do not include extra keys or text outside of this JSON structure."
        )
    else:
        system_prompt = (
            "You are a creative writing AI with expertise in literary styles."
            f"The user wants a passage written **exclusively** in a blend of the following top influences: {style_weights}."
            "Please follow these steps:"
            "1. **Identify less than 10 defining features** of the blended style strictly from these top authors."
            "   - Provide these as a concise, comma-separated list."
            f"   - Ensure the percentages match exactly with those in {style_weights} (e.g., Doyle (98.36%))."
            "   - Emphasize the influences in proportion to their weights (higher-weighted authors have a stronger impact)."
            "2. **Generate a ~200-word passage** that uses only the identified authors’ styles."
            f"   - Retain the proportions specified by {style_weights}."
            "   - If the user has provided a snippet, continue its context or tone naturally."
            "3. **Explain in 3–4 sentences** how the passage reflects these specific author influences and their numeric weights."
            "   - Keep this explanation concise (100 words or fewer)."
            "4. **Output all information** as a valid JSON object with the following structure:"
            "{"
            '  "style_elements": "comma-separated stylistic traits here",'
            '  "generated_passage": "Generated text here",'
            '  "style_explanation": "Explanation of stylistic blend here"'
            "}"
            "Do not include extra keys or text outside of this JSON structure."
        )

    # 🔹 Ensure prompt is within token limit (this was misplaced before)
    prompt = prompt[:MAX_INPUT_TOKENS]

    # 🔹 Generate text using OpenAI GPT
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Here is a passage example:{prompt}Write a passage about {topic} "}
            ],
            temperature=0.7,
            max_tokens=MAX_GENERATED_TOKENS,
            response_format={"type": "json_object"} 
        )

        # 🔹 Extract and clean JSON response
        response_content = response.choices[0].message.content.strip()

        # Remove unnecessary Markdown artifacts
        response_content = response_content.replace("```json", "").replace("```", "").strip()

        # Parse JSON safely
        response_data = json.loads(response_content)

        # Validate expected keys
        required_keys = ["style_elements", "generated_passage", "style_explanation"]
        if not all(key in response_data for key in required_keys):
            raise ValueError(f"Missing expected keys in AI response: {response_data}")

        return response_data

    except openai.BadRequestError as e:
        return {
            "style_elements": "Error: Bad request to AI.",
            "generated_passage": "Error: Bad request to AI.",
            "style_explanation": f"BadRequestError: {e}"
        }
    except openai.RateLimitError as e:
        return {
            "style_elements": "Error: Rate limit exceeded.",
            "generated_passage": "Error: Rate limit exceeded.",
            "style_explanation": f"RateLimitError: {e}. Please try again later."
        }
    except (json.JSONDecodeError, ValueError) as e:
        return {
            "style_elements": "Error: AI returned invalid JSON.",
            "generated_passage": "Error: AI returned invalid JSON.",
            "style_explanation": f"JSON Parsing Error: {e}"
        }
    except Exception as e:
        return {
            "style_elements": "Error: Unexpected issue.",
            "generated_passage": "Error: Unexpected issue.",
            "style_explanation": f"An unknown issue occurred. Details: {e}"
        }