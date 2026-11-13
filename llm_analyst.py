from dotenv import load_dotenv
from openai import OpenAI
from ml_engine import predict_price
import pandas as pd

load_dotenv()

client = OpenAI()

def generate_market_report(property_features: dict, predicted_price: float) -> str:
    prompt = f"""
    You are a senior real estate analyst. 

    Our machine learning model estimated this property at ${predicted_price:,.2f}.

    Here are the property specifications:
    - Median Neighborhood Income index: {property_features.get('MedInc')}
    - House Age: {property_features.get('HouseAge')} years
    - Average Rooms: {property_features.get('AveRooms')}
    - Average Bedrooms: {property_features.get('AveBedrms')}
    - Neighborhood Population: {property_features.get('Population')}
    - Average Occupancy: {property_features.get('AveOccup')}
    - Coordinates: ({property_features.get('Latitude')}, {property_features.get('Longitude')})
    
    Based on these details and the predicted price, generate a concise investment analysis report with three sections:
    1. Valuation Overview (explain why the price makes sense based on location and income)
    2. Pros and Cons (strengths and potential risks of this asset)
    3. Negotiation Strategy (tactical advice for the buyer)
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a professional real estate investment analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    sample_house = {
        'MedInc': 8.3252,
        'HouseAge': 41.0,
        'AveRooms': 6.98,
        'AveBedrms': 1.02,
        'Population': 322.0,
        'AveOccup': 2.55,
        'Latitude': 37.88,
        'Longitude': -122.23
    }

    input_df = pd.DataFrame([sample_house])
    estimated_price = predict_price(input_df)[0]

    report = generate_market_report(sample_house, estimated_price)
    print("--- AI Investment Report ---")
    print(report)