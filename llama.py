import csv
from langchain_ollama import OllamaLLM

# Water quality standards reference (EPA guidelines)
WATER_QUALITY_STANDARDS = {
    "pH": {"min": 6.5, "max": 8.5, "source": "EPA drinking water standards"},
    "Temperature": {"max": 29, "source": "EPA aquatic life protection (varies by ecosystem)"},
    "Turbidity": {"max": 1, "source": "EPA drinking water standards"},
    "Dissolved Oxygen": {"min": 5, "source": "EPA aquatic life protection"},
    "Conductivity": {"max": 500, "source": "Freshwater stream guidelines"}
}


def generate_prompts(row):
    prompts = []

    # Prompt 1: Structured Technical Analysis with Standards Comparison
    prompt1 = (
        f"As a senior water quality analyst, create a comprehensive report for Sample ID {row['Sample ID']} ({row['Date']}):\n"
        f"Data:\n- pH: {row['pH']}\n- Temperature: {row['Temperature (°C)']}°C\n- Turbidity: {row['Turbidity (NTU)']} NTU\n"
        f"- Dissolved Oxygen: {row['Dissolved Oxygen (mg/L)']} mg/L\n- Conductivity: {row['Conductivity (µS/cm)']} µS/cm\n"
        f"- Weather: {row['Weather predictions in F']}°F\n- Wind: {row['Wind Speed (m/s)']} m/s\n- Pressure: {row['Atmospheric Pressure (hPa)']} hPa\n\n"
        "Structure your analysis with these sections:\n"
        "1. Standards Compliance Check (compare against EPA guidelines)\n"
        "2. Parameter Interrelationships Analysis\n"
        "3. Weather Impact Assessment\n"
        "4. Anomaly Detection\n"
        "5. Environmental Risk Evaluation\n"
        "6. Management Recommendations\n"
        "Use bullet points and highlight critical findings in bold."
    )
    prompts.append(prompt1)

    # Prompt 2: Environmental Impact Scenario
    prompt2 = (
        f"Assess potential ecosystem impacts for Sample ID {row['Sample ID']}:\n"
        f"- pH: {row['pH']} | DO: {row['Dissolved Oxygen (mg/L)']} mg/L | Turbidity: {row['Turbidity (NTU)']} NTU\n"
        "Predict effects on:\n1. Fish populations\n2. Aquatic plant life\n3. Microbial communities\n"
        "4. Water treatment requirements\nInclude mitigation strategies for any identified risks."
    )
    prompts.append(prompt2)

    # Prompt 3: Weather Correlation Analysis
    prompt3 = (
        f"Analyze how weather conditions ({row['Weather predictions in F']}°F, {row['Wind Speed (m/s)']} m/s wind, "
        f"{row['Atmospheric Pressure (hPa)']} hPa) might influence the water parameters in Sample ID {row['Sample ID']}:\n"
        "Consider:\n- Thermal stratification\n- Oxygen solubility\n- Runoff potential\n- Microbial activity\n"
        "Provide both immediate effects and 48-hour predictions."
    )
    prompts.append(prompt3)

    # Prompt 4: Actionable Recommendations
    prompt4 = (
        f"Generate concrete management actions for Sample ID {row['Sample ID']} based on:\n"
        f"- pH: {row['pH']} (target 6.5-8.5)\n- Temperature: {row['Temperature (°C)']}°C (max 29°C)\n"
        f"- Turbidity: {row['Turbidity (NTU)']} NTU (max 1 NTU)\n- DO: {row['Dissolved Oxygen (mg/L)']} mg/L (min 5 mg/L)\n"
        "Format as:\n1. Priority Level\n2. Parameter\n3. Recommended Action\n4. Expected Outcome\n"
        "Include chemical treatment options if needed."
    )
    prompts.append(prompt4)

    # Prompt 5: Historical Trend Hypothesis
    prompt5 = (
        f"Based on current measurements from {row['Date']} (Sample ID {row['Sample ID']}):\n"
        f"- Conductivity: {row['Conductivity (µS/cm)']} µS/cm\n- Temperature: {row['Temperature (°C)']}°C\n"
        f"- Weather: {row['Weather predictions in F']}°F\n"
        "Predict likely trends for:\n1. Next 24 hours\n2. Next 7 days\n"
        "Consider seasonal patterns and provide confidence estimates for predictions."
    )
    prompts.append(prompt5)

    return prompts


def main():
    llm = OllamaLLM(
        model="llama3.1",
        temperature=0.3,  # Reduced for more factual responses
        top_p=0.9,
        num_ctx=4096,  # Longer context window
        system="You are a water quality expert with 20 years of experience. "
               "Your responses should be technical yet accessible, citing environmental science principles "
               "and EPA regulations. Always provide actionable recommendations and quantify risks when possible."
    )

    csv_file_path = '/Users/admin/Downloads/Water Quality Testing w forecast.csv'

    with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            prompts = generate_prompts(row)

            # Process prompts with improved chunking
            for i, prompt in enumerate(prompts, 1):
                response = llm.invoke([prompt])  # Process one prompt at a time
                print(f"Sample ID {row['Sample ID']} - Analysis {i}:")
                print(response)
                print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    main()