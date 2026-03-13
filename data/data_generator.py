import os
import json
import time
import random
import re # Remember to import re
from dotenv import load_dotenv
from tqdm import tqdm

# --- NEW IMPORTS ---
from google import genai
from google.genai import types

# --- 1. CONFIGURATION ---
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Missing GEMINI_API_KEY")

# Initialize Client (Replaces genai.configure)
client = genai.Client(api_key=api_key)

# Safety & Generation Configuration (using types from new SDK)
generate_config = types.GenerateContentConfig(
    temperature=0.85,
    top_p=0.95,
    max_output_tokens=8192,
    response_mime_type="application/json",
    safety_settings=[
        types.SafetySetting(
            category="HARM_CATEGORY_HARASSMENT",
            threshold="BLOCK_NONE"
        ),
        types.SafetySetting(
            category="HARM_CATEGORY_HATE_SPEECH",
            threshold="BLOCK_NONE"
        ),
        types.SafetySetting(
            category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
            threshold="BLOCK_NONE"
        ),
        types.SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold="BLOCK_NONE"
        ),
    ]
)

# Model Name
MODEL_NAME = "gemini-2.5-flash" # Hoặc gemini-1.5-flash

# Define valid labels for validation
VALID_LABELS = {
    "O", 
    "B-ROLE", "I-ROLE", 
    "B-SKILL", "I-SKILL", 
    "B-LOC", "I-LOC", 
    "B-EXP", "I-EXP", 
    "B-SALARY", "I-SALARY"
}

generation_config = {
    "temperature": 0.85, 
    "top_p": 0.95,
    "max_output_tokens": 8192, 
    "response_mime_type": "application/json",
}


CONTEXT_DATA = {
    "Modern Web Fullstack": {
        "roles": ["Fullstack Developer", "Frontend Engineer", "Backend Node.js Dev", "Technical Lead", "Software Architect"],
        "skills": ["React.js", "Next.js", "Vue.js", "Node.js", "NestJS", "TypeScript", "Tailwind CSS", "GraphQL", "MongoDB", "Redis", "Microservices", "RESTful API"]
    },
    "Java & Enterprise Backend": {
        "roles": ["Java Backend Engineer", "Spring Boot Developer", "J2EE Specialist", "Software Engineer (Java)", "Microservices Architect"],
        "skills": ["Java 17", "Spring Boot", "Hibernate", "Kafka", "RabbitMQ", "Oracle DB", "Microservices Design", "Docker", "Maven", "Gradle", "PostgreSQL"]
    },
    "Mobile & Cross-platform": {
        "roles": ["iOS Developer", "Android Engineer", "Flutter Dev", "React Native Expert", "Mobile Architect"],
        "skills": ["Swift", "Kotlin", "Dart", "Objective-C", "Jetpack Compose", "Xcode", "Firebase SDK", "SwiftUI", "Android SDK", "Redux Mobile", "CocoaPods"]
    },
    "PHP & CMS Development": {
        "roles": ["PHP Developer", "WordPress Developer", "Magento Specialist", "Laravel Developer", "Backend Engineer"],
        "skills": ["PHP 8", "Laravel", "WordPress Core", "Magento 2", "MySQL", "Symfony", "Drupal", "WooCommerce", "Plugin Development", "HTML/CSS"]
    },
    "Ruby & Scripting Niche": {
        "roles": ["Ruby on Rails Developer", "Python Backend Dev", "Perl Scripter", "Backend Engineer (Ruby)"],
        "skills": ["Ruby on Rails", "RSpec", "Sidekiq", "PostgreSQL", "Redis", "Heroku", "Capistrano", "Django", "Flask", "FastAPI"]
    },
    "Golang & High Performance": {
        "roles": ["Golang Engineer", "Backend Developer (Go)", "Distributed Systems Engineer"],
        "skills": ["Golang (Go)", "gRPC", "Protobuf", "Concurrency", "Microservices", "CockroachDB", "NATS", "Elasticsearch"]
    },
    "AI & Data Science": {
        "roles": ["AI Researcher", "NLP Engineer", "Computer Vision Specialist", "MLOps Engineer", "Data Scientist", "Prompt Engineer"],
        "skills": ["PyTorch", "HuggingFace Transformers", "LangChain", "OpenCV", "TensorFlow", "RAG", "MLflow", "Pandas", "CUDA", "LlamaIndex", "Scikit-learn", "Vector Database"]
    },
    "Data Engineering": {
        "roles": ["Data Engineer", "ETL Developer", "Big Data Engineer", "Analytics Engineer", "BI Developer"],
        "skills": ["Apache Spark", "Hadoop", "Kafka", "Airflow", "Snowflake", "Databricks", "dbt", "SQL Server", "Tableau", "PowerBI", "Data Warehousing", "Python ETL"]
    },
    "Database Administration (DBA)": {
        "roles": ["Database Administrator", "Oracle DBA", "PostgreSQL Expert", "MySQL Database Engineer", "NoSQL Specialist"],
        "skills": ["Oracle RAC", "PL/SQL", "PostgreSQL Tuning", "MySQL Replication", "MongoDB Sharding", "Cassandra", "Backup & Recovery", "Database Security", "SQL Optimization"]
    },
    "Cloud & DevOps": {
        "roles": ["DevOps Engineer", "Site Reliability Engineer (SRE)", "Cloud Architect", "Platform Engineer", "Azure Administrator"],
        "skills": ["Kubernetes", "Terraform", "AWS Lambda", "Docker Swarm", "CI/CD Jenkins", "Ansible", "Prometheus", "Grafana", "Azure Bicep", "Helm", "CircleCI", "Linux Fundamentals"]
    },
    "Network & System Engineering": {
        "roles": ["Network Engineer", "System Administrator", "Network Architect", "Cisco Specialist", "IT Infrastructure Manager"],
        "skills": ["Cisco CCNA/CCNP", "Juniper", "BGP/OSPF", "Firewall (Palo Alto/Fortinet)", "VPN", "Active Directory", "Windows Server", "VMware vSphere", "DNS/DHCP", "Load Balancing"]
    },
    "Cybersecurity": {
        "roles": ["Penetration Tester", "Security Analyst", "SOC Analyst", "Ethical Hacker", "AppSec Engineer", "CISO"],
        "skills": ["Metasploit", "Wireshark", "Burp Suite", "OWASP Top 10", "SIEM Splunk", "Kali Linux", "Reverse Engineering", "CISSP", "Network Security", "Cryptography", "ISO 27001"]
    },
    "Product & Project Management": {
        "roles": ["Product Owner", "Product Manager", "Scrum Master", "Business Analyst (BA)", "Project Manager", "Agile Coach"],
        "skills": ["Agile/Scrum", "JIRA", "User Stories", "Product Roadmap", "Stakeholder Management", "Kanban", "SQL for Analysis", "A/B Testing", "UML Diagrams", "PMP"]
    },
    "UI/UX & Design": {
        "roles": ["UI/UX Designer", "Product Designer", "Web Designer", "Graphic Designer", "Creative Director"],
        "skills": ["Figma", "Adobe XD", "Sketch", "Wireframing", "Prototyping", "User Research", "Adobe Creative Cloud", "Material Design", "HTML/CSS Basics", "Interaction Design"]
    },
    "ERP, CRM & Salesforce": {
        "roles": ["Salesforce Developer", "Dynamics 365 Consultant", "ERP Specialist", "SAP Consultant", "NetSuite Developer"],
        "skills": ["Salesforce Apex", "Lightning Web Components", "SAP ABAP", "Microsoft Dynamics 365", "Odoo", "CRM Configuration", "Business Process Mapping", "Oracle EBS"]
    },
    "Game Development": {
        "roles": ["Game Developer", "Unity Developer", "Unreal Engine Programmer", "Gameplay Engineer", "Technical Artist"],
        "skills": ["Unity 3D", "C#", "Unreal Engine 5", "C++", "DirectX", "OpenGL", "Shader Graph", "Blender", "Physics Engines", "Multiplayer Networking"]
    },
    "Embedded & IoT": {
        "roles": ["Embedded Software Engineer", "Firmware Engineer", "IoT Developer", "Hardware Design Engineer", "Automotive Engineer"],
        "skills": ["C/C++", "RTOS", "Microcontrollers (STM32/ESP32)", "UART/I2C/SPI", "Altium Designer", "Verilog", "Embedded Linux", "Raspberry Pi", "AUTOSAR", "CAN Bus"]
    },
    "Blockchain & Web3": {
        "roles": ["Smart Contract Developer", "Solidity Engineer", "Rust Protocol Dev", "Blockchain Architect"],
        "skills": ["Solidity", "Hardhat", "Web3.js", "Rust", "Hyperledger Fabric", "Truffle", "Smart Contracts", "DeFi", "Ethers.js", "Cryptography"]
    },
    "Robotics & Automation": {
        "roles": ["Robotics Engineer", "Control Systems Engineer", "Automation Engineer (PLC)", "RPA Developer"],
        "skills": ["ROS (Robot Operating System)", "SLAM", "C++", "PLC Programming (Siemens/Allen Bradley)", "UiPath", "Computer Vision", "MATLAB/Simulink", "Industrial Automation"]
    },

    "IT Support & Helpdesk": {
        "roles": ["IT Support Specialist", "Helpdesk Technician", "IT Operations Officer", "Technical Support Engineer"],
        "skills": ["Troubleshooting", "Office 365 Admin", "Hardware Repair", "Remote Desktop Tools", "Ticket Systems (ServiceNow/Jira Service Desk)", "Printer Configuration", "Customer Service", "MacOS/Windows Support"]
    },
    "QA & Automation Testing": {
        "roles": ["Automation Tester", "QA Lead", "Manual Tester", "QC Engineer", "SDET"],
        "skills": ["Selenium WebDriver", "Appium", "Postman", "JMeter", "Cucumber", "TestNG", "Katalon Studio", "Bug Tracking", "API Testing", "Playwright", "Cypress"]
    }
}
STYLES = [
    # --- BASIC & PROFESSIONAL GROUP ---
    # 1. Urgent Headhunter
    "Headhunter Urgent: Use uppercase for keywords, focus on high salary/bonuses, short sentences, sense of urgency (ASAP, Urgent).",
    
    # 2. Serious Tech Lead
    "Tech Lead Requirements: Dry, technical tone. List specific versions (Java 17, .NET 6). Focus on hard skills, minimal fluff.",
    
    # 3. Formal Corporate
    "Corporate Formal: Professional, long sentences, standard grammar. Focus on responsibilities, 'ideal candidate' descriptions.",

    # 4. Government/Bank/High Security (NEW)
    "Government/Defense/Banking: Extremely formal, rigid requirements. Mention 'Security Clearance', 'Citizenship', 'Compliance', 'ISO standards'. Use legacy terms.",

    # --- INFORMAL GROUP (DIRTY DATA) ---
    # 5. Friendly Startup
    "Startup Casual: Friendly tone, use emojis, flexible requirements, mentions 'beer', 'snacks', 'ping pong'.",
    
    # 6. Bullet Points (Super short)
    "Bullet Points: Start sentences with verbs or just keywords. Extremely concise. No 'We are looking for'. Example: '- Strong Java exp', '* Python/Django needed'",
    
    # 7. Messy/Chat/Messages (Important)
    "Messy/Casual: Lowercase, lazy typing, ignore grammar, use abbreviations like 'exp', 'yrs', 'k', 'dev', 'pls'. Example: 'need java dev 5yrs exp, salary 2k$'. Typo example: 'Phython' instead of 'Python'.",

    # 8. Broken English (Non-native)
    "Broken English: Simulate non-native speakers. Wrong prepositions, missing articles, awkward phrasing. Example: 'We finding developer expert Java', 'Company have good benefit'.",

    # 9. Social Media Style (Facebook/Twitter/LinkedIn) (NEW)
    "Social Media Post: Use hashtags (#hiring #job), call to actions ('DM me', 'Link in bio'), lots of emojis, informal greetings ('Hey guys', 'Folks').",

    # --- SPECIFIC TARGET GROUP ---
    # 10. Super short summary
    "Brief/Summary: Telegraphic style. Bullet-point like but in one line. Extremely concise (e.g., 'Java, 5y exp, Hanoi').",

    # 11. Hiring Fresher/Intern
    "Internship/Fresher: Focus on GPA, University degree, 'willingness to learn', 'passion for coding'. Low experience requirements (0-1 year). Mention 'training provided'.",

    # 12. Hiring C-Level/Executive (NEW)
    "Executive/C-Level: Focus on 'Strategy', 'Vision', 'Leadership', 'Stakeholder Management', 'Equity/Shares'. Less technical detail, more business impact.",

    # 13. Freelance/Project-based (NEW)
    "Freelance/Gig: Focus on 'Project scope', 'Hourly rate', 'Deliverables', 'Contract duration', 'Immediate start'. Specific short-term tasks.",

    # --- SPECIFIC & MARKETING GROUP ---
    # 14. "Ninja/Rockstar" style (Marketing buzzwords)
    "Buzzword/Marketing: Use terms like 'Rockstar', 'Ninja', 'Wizard', '10x Engineer'. Focus on passion, culture, 'change the world'. Vague technical requirements.",

    # 15. Hiring Remote/Global
    "Remote/Global: Focus on 'Async work', 'Timezone overlap', 'English communication', 'Self-discipline'. Mention tools like Slack, Zoom, Jira.",

    # 16. Web3/Crypto Degen (NEW - Very specific)
    "Web3/Crypto Native: Use community slang (WAGMI, LFG, Degen). Focus on 'Discord', 'DAO', 'Tokenomics', 'Whitepaper'. Very casual but high-tech.",

    # 17. "All-in-one" hiring style (NEW - Unrealistic)
    "Unrealistic/All-in-One: Ask for a Fullstack Dev who also knows DevOps, AI, Mobile, and Photoshop. 'Must know everything'. High requirements, usually vague salary.",

    # 18. Stealth Mode (NEW)
    "Stealth Mode: Vague about product details. Uses phrases like 'Well-funded startup', 'Disruptive technology', 'NDA required', 'Next big thing'.",

    # 19. Internal recruiting/Referral (NEW)
    "Internal/Referral: Conversational tone between colleagues. 'My team needs a...', 'Let me know if you know anyone'.",

    # 20. Q&A Style (NEW)
    "Q&A Style: Written as questions. 'Do you love React?', 'Are you tired of legacy code?', 'Want to work on high scale systems?'."
]

# --- 3. PROMPT GENERATOR (ENGLISH ONLY) ---
def generate_prompt(domain, context_info, style, batch_size=20):
    roles = ", ".join(context_info['roles'])
    skills = ", ".join(context_info['skills'])
    
    return f"""
    You are an expert Data Synthesizer for NLP. Your task is to generate high-quality NER (Named Entity Recognition) training data for the IT Recruitment domain.
    
    ### 1. INPUT CONTEXT
    - **Domain:** {domain}
    - **Target Roles:** {roles}
    - **Key Skills:** {skills}
    - **Writing Style:** {style}
    
    ### 2. ENTITY DEFINITIONS (STRICT)
    - **ROLE:** Job titles (e.g., "Senior Backend Engineer", "DevOps Lead"). Include seniority (Senior, Junior, Lead) inside the tag.
    - **SKILL:** Technical skills, tools, frameworks, languages (e.g., "Java", "AWS", "CI/CD", "Agile"). 
    - **LOC:** Locations or Work modes (e.g., "New York", "Berlin", "Remote", "Hybrid", "On-site").
    - **EXP:** Years of experience or level requirements (e.g., "5 years", "5+ yrs", "Junior level").
    - **SALARY:** Monetary values, specific benefits logic (e.g., "$2000", "150k", "competitive salary").
    - **O:** Outside/Irrelevant tokens.

    ### 3. CRITICAL TOKENIZATION RULES (MUST FOLLOW)
    1. **Punctuation Split:** Every punctuation mark (., ,, -, /, &) must be a separate token.
       - Bad: ["Node.js", "C/C++"]
       - Good: ["Node", ".", "js", "C", "/", "C++"]
    2. **Currency Split:** Currency symbols must be separate tokens.
       - Bad: ["$150k"]
       - Good: ["$", "150", "k"]
    3. **Range Split:** Ranges must be split.
       - Bad: ["3-5"]
       - Good: ["3", "-", "5"]
    4. **Alignment:** The length of `tokens` list MUST equal the length of `ner_tags` list.

    ### 4. GENERATION TASK
    Generate exactly **{batch_size} unique samples** in a raw JSON list format.
    
    **Distribution Requirements:**
    - 20% Short/Headlines (e.g., "Senior Java Dev - $120k - Remote")
    - 50% Medium/Requirements (Standard sentences)
    - 30% Long/Benefits/Complex (Full paragraphs)
    
    **Special Instruction for Negative Samples:** - Include 2-3 sentences where tech keywords appear but represent hardware/perks, NOT skills. Label them as 'O'.
    - Example: "Free MacBook Pro M1 provided." -> "MacBook", "Pro", "M1" are 'O', not 'SKILL'.

    ### 5. EXAMPLE OUTPUT (Observe Tokenization & Tags)
    [
      {{
        "tokens": ["We", "need", "a", "Senior", "Java", "Developer", "with", "5", "+", "years", "exp", "in", "Spring", "Boot", "."],
        "ner_tags": ["O", "O", "O", "B-ROLE", "I-ROLE", "I-ROLE", "O", "B-EXP", "I-EXP", "I-EXP", "I-EXP", "O", "B-SKILL", "I-SKILL", "O"]
      }},
      {{
        "tokens": ["Salary", "range", ":", "$", "100", "k", "-", "$", "140", "k", "/", "year", "."],
        "ner_tags": ["O", "O", "O", "B-SALARY", "I-SALARY", "I-SALARY", "I-SALARY", "I-SALARY", "I-SALARY", "I-SALARY", "I-SALARY", "I-SALARY", "O"]
      }},
      {{
        "tokens": ["Benefit", ":", "New", "MacBook", "Pro", "and", "iPhone", "14", "."],
        "ner_tags": ["O", "O", "O", "O", "O", "O", "O", "O", "O"]
      }}
    ]

    ### OUTPUT
    Return **ONLY** the JSON list. Do not use Markdown code blocks (```json). Do not explain.
    """
# --- 4. VALIDATION & UTILS ---
def clean_json_string(text):
    """
    Clean up JSON string returned from Gemini (remove Markdown ```json ... ```).
    """
    if not text: return None
    
    # Find JSON segment within list [...]
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        return match.group(0)
    return text

def validate_data(tokens, tags):
    """Check data consistency and validity (Strict Mode)"""
    # 1. Check length
    if len(tokens) != len(tags) or len(tokens) < 3:
        return False
    
    # 2. (NEW) Check for empty tokens or newline characters
    if any(not t.strip() for t in tokens): return False 
    if any("\n" in t for t in tokens): return False
    
    # 3. Check BIO logic
    for i, tag in enumerate(tags):
        if tag not in VALID_LABELS:
            return False
            
        if tag.startswith("I-"):
            if i == 0: return False
            prev_tag = tags[i-1]
            current_type = tag.split("-")[1]
            
            if prev_tag == "O": return False
            if prev_tag.split("-")[1] != current_type: return False
            
    return True

def load_existing_data(filepath):
    """Load existing data to avoid duplicates"""
    seen_hashes = set()
    count = 0
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # Create hash based on tokens string
                    content_hash = hash(tuple(data['tokens']))
                    seen_hashes.add(content_hash)
                    count += 1
                except:
                    continue
    return seen_hashes, count

def generate_with_retry(prompt):
    """
    Gọi API với cơ chế:
    - Nếu hết Quota (429): Chờ 60s rồi thử lại (Lặp vô hạn).
    - Nếu lỗi khác (Mạng, 500): Thử lại 5 lần rồi bỏ qua.
    """
    max_retries = 5
    base_wait = 10 
    
    while True: # Infinite loop to handle Quota
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=MODEL_NAME, 
                    contents=prompt, 
                    config=generate_config
                )

                if not response.text:
                    print(f"Empty Response. Retrying {attempt+1}/{max_retries}")
                    continue
                    
                return response.text

            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "ResourceExhausted" in error_msg:
                    print(f"Rate limit (429). Sleeping 60s...")
                    time.sleep(60)
                    break 
                

                print(f"API Error ({attempt+1}/{max_retries}): {e}")
                time.sleep(base_wait)
        
        if "429" in error_msg or "ResourceExhausted" in error_msg:
            continu
        else:
            return None

# --- MAIN FUNCTION ---
def main():
    # --- CONFIGURATION ---
    TARGET_SAMPLES = 10000    
    BATCH_SIZE = 10          
    OUTPUT_FILE = "dataset.jsonl"
    
    # Load existing data
    seen_hashes, current_count = load_existing_data(OUTPUT_FILE)
    print("--- STATUS ---")
    print(f"Existing samples: {current_count}")
    
    if current_count >= TARGET_SAMPLES:
        print("Target samples reached. Stopping program.")
        return

    needed = TARGET_SAMPLES - current_count
    print(f"Need to generate: {needed} samples.")
    
    pbar = tqdm(total=needed, desc="Generating")
    consecutive_errors = 0 
    
    while current_count < TARGET_SAMPLES:
        if consecutive_errors > 10:
            print("Too many consecutive errors. Stopping.")
            break

        domain = random.choice(list(CONTEXT_DATA.keys()))
        context_info = CONTEXT_DATA[domain]
        style = random.choice(STYLES)
        
        prompt = generate_prompt(domain, context_info, style, batch_size=BATCH_SIZE)
        
        # Call API
        raw_text = generate_with_retry(prompt)
        
        if not raw_text: 
            consecutive_errors += 1
            continue
        
        try:
            clean_text = clean_json_string(raw_text)
            if not clean_text: raise ValueError("Empty JSON after clean")

            json_output = json.loads(clean_text)
            if isinstance(json_output, dict): json_output = [json_output]
            
            valid_batch = []
            
            for sample in json_output:
                tokens = sample.get("tokens", [])
                tags = sample.get("ner_tags", [])
                
                if not validate_data(tokens, tags): continue
                
                content_str = json.dumps(tokens)
                if content_str in seen_hashes: continue
                
                seen_hashes.add(content_str)
                valid_batch.append(sample)
            
            if valid_batch:
                with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                    for item in valid_batch:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                
                added_count = len(valid_batch)
                current_count += added_count
                pbar.update(added_count)
                consecutive_errors = 0 
            
            time.sleep(2) 
            
        except Exception as e:
            # print(f"Error: {e}")
            consecutive_errors += 1

    pbar.close()
    print(f"\nCompleted! Total samples: {current_count}")

# Cần định nghĩa lại các hàm phụ trợ (validate_data, clean_json_string, load_existing_data) 
# giống hệt code trước đó tôi gửi, chỉ thay đổi phần gọi API thôi.
if __name__ == "__main__":
    main()