import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from xgboost import XGBClassifier
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge
import openai

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# 1. Data Cleaning Function
def data_cleaning(df):
    pattern = re.compile(r'[^a-zA-Z\s]')
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    def clean_text(text):
        text = pattern.sub('', text).lower().strip()
        tokens = nltk.word_tokenize(text)
        tokens = [lemmatizer.lemmatize(word)
                  for word in tokens if word not in stop_words]
        return ' '.join(tokens)

    df['text'] = df['text'].apply(clean_text)
    return df


# 2. Select Top Helpful Reviews
def get_top_reviews(group):
    low_rated = group[group['rating'] <= 2].nlargest(2, 'helpful_vote')
    high_rated = group[group['rating'] >= 4].nlargest(2, 'helpful_vote')
    return pd.concat([low_rated, high_rated])


# 3. LLM Invocation for Synthetic Reviews
def invoke_llm(prompt):
    client = OpenAI(
        base_url="BASE_URL",
        api_key="API_KEY"
    )

    completion = client.Completion.create(
        model="nvidia/nemotron-4-340b-instruct",
        prompt=prompt,
        temperature=0.2,
        top_p=0.7,
        max_tokens=4096
    )
    return completion.choices[0].text


# 4. Prompt Generation for LLM
def gen_prompt(df):
    unique_titles = df['title_x'].unique()
    products = []

    for _ in range(3):
        random_title = random.choice(unique_titles)
        random_reviews = df[df['title_x'] == random_title]
        random_reviews.loc[:, 'text'] = random_reviews['text'].str.strip()
        random_reviews = random_reviews[random_reviews['text'].str.len() > 100]

        if random_reviews.shape[0] >= 3:
            products.append(random_reviews.head(6))

    prompt = f"""
You are acting as a synthetic review generator for healthcare, health supplement products.
Following is an example of some reviews for 3 products, with how many people thought it was helpful and rating of the product on scale of (1-5) with 1 being worst to 5 being best:

"""

    for i, product in enumerate(products, 1):
        prompt += f"Product {i}: {product.iloc[0, 0]}\n"
        for index, row in product.iterrows():
            prompt += f"{index+1}. '{row['text']}' with helpful votes: {row['helpful_vote']} and rating: {row['rating']}\n"

    prompt += """

Based on the above examples generate 25 reviews for various other health products, the review being as helpful as possible and on varied rating scale.
Output format should be a CSV having a review and a rating about sentiment of the review.
    """
    return prompt


# 5. Paraphrasing Function
def paraphrase(text):
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    tokenizer = T5Tokenizer.from_pretrained('t5-small')

    inputs = tokenizer.encode(
        f"paraphrase: {text}", return_tensors='pt', max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# 6. Generate Synthetic Reviews
def generate_synthetic_reviews(df, n=4):
    columns = ['review', 'rating']
    llm_df = pd.DataFrame(columns=columns)

    for _ in range(n):
        prompt = gen_prompt(df)
        response = invoke_llm(prompt)
        for line in response.strip().split('\n'):
            if ',' in line:
                last_comma_index = line.rfind(',')
                row_data = [line[:last_comma_index],
                            line[last_comma_index + 1:]]
                new_row = pd.DataFrame([row_data], columns=columns)
                llm_df = pd.concat([llm_df, new_row], ignore_index=True)

    return llm_df


# 7. Adversarial Validation
def adversarial_validation(df, df_synthetic):
    df_synthetic['label'] = 1
    df['label'] = 0

    combined_df = pd.concat(
        [df[['reviews', 'label']], df_synthetic[['reviews', 'label']]])
    X_train, X_test, y_train, y_test = train_test_split(
        combined_df['reviews'], combined_df['label'], test_size=0.2)

    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train.values.astype('U'))
    X_test_vec = vectorizer.transform(X_test)

    model = XGBClassifier()
    model.fit(X_train_vec, y_train)
    accuracy = model.score(X_test_vec, y_test)
    return accuracy


# 8. BLEU and ROUGE Scores Calculation
def calculate_bleu_rouge(original_reviews, synthetic_reviews):
    original_tokens = [review.split()
                       for review in original_reviews if isinstance(review, str)]
    synthetic_tokens = [review.split() for review in synthetic_reviews]

    # BLEU score
    bleu_score = corpus_bleu([[ref]
                             for ref in original_tokens], synthetic_tokens)

    # ROUGE score
    rouge = Rouge()
    rouge_score = rouge.get_scores(
        synthetic_reviews, original_reviews, avg=True)

    return bleu_score, rouge_score


# Main Execution Pipeline
def main():
    # 1. Preprocessing the data
    products = pd.read_csv("product_asin.csv")
    reviews = pd.read_csv("reviews_supplements.csv")
    merged_reviews = pd.merge(
        products, reviews, on='parent_asin', how='inner').dropna()

    merged_reviews = data_cleaning(merged_reviews)

    # 2. Generate Embeddings and Clustering
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = model.encode(merged_reviews['text'].tolist())
    kmeans = KMeans(n_clusters=5, random_state=0).fit(embeddings)
    merged_reviews['cluster'] = kmeans.labels_

    # 3. Reduce Data Size for Efficient Processing
    result = merged_reviews.groupby('title_x').apply(
        get_top_reviews).reset_index(drop=True)
    result.to_csv("merged_reviews_small.csv", index=False)

    # 4. Generate Synthetic Reviews
    llm_df = generate_synthetic_reviews(result)
    llm_df.to_csv("synthetic_data.csv", index=False)

    # 5. Paraphrase Reviews (Optional)
    llm_df['paraphrased_review'] = llm_df['review'].apply(paraphrase)
    llm_df.to_csv("postprocess.csv", index=False)

    # 6. Adversarial Validation
    accuracy = adversarial_validation(result[['text']], llm_df[['review']])
    print(f"Adversarial Validation Accuracy: {accuracy}")

    # 7. BLEU and ROUGE Scores
    original_reviews = result['text'].tolist()
    synthetic_reviews = llm_df['review'].tolist()
    bleu_score, rouge_score = calculate_bleu_rouge(
        original_reviews, synthetic_reviews)
    print(f"BLEU Score: {bleu_score}")
    print(f"ROUGE Score: {rouge_score}")


if __name__ == "__main__":
    main()
