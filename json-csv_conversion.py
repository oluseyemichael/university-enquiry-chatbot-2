import json
import pandas as pd

# Load the JSON data
data = '''
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": [
        "Hi",
        "Greetings",
        "Hey",
        "How are you",
        "Is anyone there?",
        "Hello",
        "Good day"
      ],
      "responses": [
        "Hey :-)",
        "Hello, thanks for visiting",
        "Hi there, what can I do for you?",
        "Hi there, how can I help?",
        "Hi, welcome to AJU, how can i help?"
      ]
    },
    {
      "tag": "goodbye",
      "patterns": [
        "Bye",
        "See you later",
        "Goodbye"
      ],
      "responses": [
        "See you later, thanks for visiting",
        "Have a nice day",
        "Bye!."
      ]
    },
    {
      "tag": "thanks",
      "patterns": [
        "Thanks",
        "Thank you",
        "That's helpful",
        "Thank's a lot!"
      ],
      "responses": [
        "Happy to help!",
        "Any time!",
        "My pleasure"
      ]
    },
    {
      "tag": "general_information",
      "patterns": [
        "Tell me about the history of the university.",
        "What are some interesting facts about the university?",
        "Who are you guys?",
        "Tell me about the school?"
      ],
      "responses": [
        "Arrupe Jesuit University (AJU) is an autonomous international academic institution, owned by the Southern Africa Province of the Society of Jesus on the behalf of the Jesuit Conference of Africa and Madagascar (JCAM). Click here to view the <a href='https://www.aju.ac.zw/our-history/'>history of the school</a>. View our <a href='https://www.aju.ac.zw/'>Home Page here</a>"
      ]
    },
    {
      "tag": "name",
      "patterns": [
        "What is your name?"
      ],
      "responses": [
        "My name is AJU Assistant and i'm here to help with your queries."
      ]
    },
    {
      "tag": "library",
      "patterns": [
        "Is there a library on campus?",
        "Do you have a library?",
        "Where is the library?"
      ],
      "responses": [
        "There is a library on campus that houses a vast collection of books. View more library information <a href= 'https://www.aju.ac.zw/our-library/'>here</a>."
      ]
    },
    {
      "tag": "facilities",
      "patterns": [
        "What are the facilities available at the university?",
        "what facilities does the school have?"
      ],
      "responses": [
        "Facilities at the university include an ICT/Computer Lab, Sports facilites, Library and a Gym."
      ]
    },
    {
      "tag": "campus_information",
      "patterns": [
        "What about campus?",
        "How is campus life?"
      ],
      "responses": [
        "Campus life is rich in culture. Click <a href= 'https://www.aju.ac.zw/accomodation/'>here</a> for more information"
      ]
    },
    {
      "tag": "open_information",
      "patterns": [
        "Do i have to be a Jesuit to join?",
        "Do you accept girls?",
        "Can anybody come?"
      ],
      "responses": [
        "AJU is open to everyone"
      ]
    },
    {
      "tag": "accommodation",
      "patterns": [
        "Do you have accommodation for students?",
        "Do you have houses for students?",
        "Where do students stay?",
        "How much for accommodation?"
      ],
      "responses": [
        "View the details of accommodation <a href = https://www.aju.ac.zw/accomodation/>here</a>"
      ]
    },
    {
      "tag": "location",
      "patterns": [
        "Where are you guys?",
        "Where is the university located?",
        "Where is the school?",
        "location",
        "where are you located?",
        "where is campus?"
      ],
      "responses": [
        "The university is located in No. 16 Link Road, P.O Box MP 320 Mount Pleasant, Zimbabwe. View on <a href='https://www.google.com/maps/place/Arrupe+Jesuit/@-17.7761599,31.0558872,17z/data=!3m1!4b1!4m6!3m5!1s0x1931af8ad1f3dc95:0xd7d003b823669882!8m2!3d-17.776165!4d31.0584621!16s%2Fg%2F11bxfv7ccr?entry=ttu'>Maps</a>"
      ]
    },
    
    {
      "tag": "courses_information",
      "patterns": [
        "What programs does the university offer?",
        "Tell me about the academic programs at the university.",
        "What courses do you offer?"
      ],
      "responses": [
        "The university offers a variety of programs including - BA Honors Degree in Philosophy, Masters Degree in Philosophy, Cisco Networking Academy, BSc (Honours) in Information & Communication Technology, BA Honours in Transformational Leadership."
      ]
    },
    {
      "tag": "departments_information",
      "patterns": [
        "What are the departments in the school?"
      ],
      "responses": [
        "The departments in the school are - SCHOOL OF PHILOSOPHY & HUMANITIES, SCHOOL OF ENGINEERING & ICT, SCHOOL OF EDUCATION & LEADERSHIP, SCHOOL OF HEALTH SCIENCE AND TECHNOLOGY, SCHOOL OF MANAGEMENT AND DEVELOPMENT DEPARTMENT OF LANGUAGES."
      ]
    },
    {
      "tag": "courses_duration",
      "patterns": [
        "How long is the duration of ICT Program?"
      ],
      "responses": [
        "The departments in the school are - SCHOOL OF PHILOSOPHY & HUMANITIES, SCHOOL OF ENGINEERING & ICT, SCHOOL OF EDUCATION & LEADERSHIP, SCHOOL OF HEALTH SCIENCE AND TECHNOLOGY, SCHOOL OF MANAGEMENT AND DEVELOPMENT DEPARTMENT OF LANGUAGES."
      ]
    },
    {
      "tag": "general_admissions",
      "patterns": [
        "How can I apply for admission?",
        "What are the application deadlines?",
        "What are the admission requirements?",
        "Is there an entrance exam for admission?",
        "Can I change my major after admission?",
        "How do local students apply?",
        "How can i apply?",
        "How do i register?"
      ],
      "responses": [
        "You can visit the university's official admissions portal for admission information. Click <a href='https://www.aju.ac.zw/aju-applications/'>here</a> "
      ]
    },
    {
      "tag": "intl_admissions",
      "patterns": [
        "What are the admission requirements for international students?",
        "What are the admission requirements for foreign students?",
        "Do you accept foreginers?",
        "How do foreiners apply?",
        "Can i apply from a different country?"
      ],
      "responses": [
        "Click on this <a href='https://www.aju.ac.zw/2020/12/31/international-students/'>link</a> to view Admission requirements for international students"
      ]
    },
    {
      "tag": "fees",
      "patterns": [
        "How much is the tuition fee for ICT?",
        "What is the payment schedule for tuition?",
        "Are there additional fees for international students?",
        "How much is the school fees?",
        "How do i pay the fees?",
        "How much is school fees?"
      ],
      "responses": [
        "For all school fees information please click <a href= 'https://www.aju.ac.zw/tuition-fees/'>here</a>  "
      ]
    },
    {
      "tag": "scholarship",
      "patterns": [
        "How can i get a scholarship?",
        "Are scholarships available?",
        "Are there any scholarships available?",
        "How do i apply for a scholarship",
        "How do i get a scholarship?"
      ],
      "responses": [
        "Scholarships are available for eligible students. You can check by clicking <a href='https://www.aju.ac.zw/scholarships/'>here</a>"
      ]
    },
    {
      "tag": "timetable",
      "patterns": [
        "Where can I find the class timetable?",
        "timetable",
        "How can I check my course schedule?",
        "Where is the timetable?"
      ],
      "responses": [
        "The class timetable is available on the JCAM Portal. Login <a href='https://jesuits-africa.education/acis/ais/'>here</a>  "
      ]
    },
    {
      "tag": "exam_schedule",
      "patterns": [
        "When is the exam?",
        "What date is the exam?",
        "When is the next exam starting?",
        "When is the next exam scheduled?"
      ],
      "responses": [
        "Exam schedules are typically posted on the notice board and the student portal. Please check for updates."
      ]
    },
    {
      "tag": "feedback",
      "patterns": [
        "Where can I submit feedback about a course?"
      ],
      "responses": [
        "Feedback about a course can be submitted through the course evaluation system."
      ]
    },
    {
      "tag": "contact_information",
      "patterns": [
        "How can I contact the admissions office?",
        "How do i call?",
        "How do i contact?",
        "What is your contact information"
      ],
      "responses": [
        "You can contact the admissions office by email - admissions@aju.ac.zw , or by phone - +263 242 745411"
      ]
    }
  ]
}
'''

# Parse the JSON data
parsed_data = json.loads(data)

# Extract the intents list
intents = parsed_data['intents']

# Convert to DataFrame
df = pd.DataFrame(intents)

# Save to CSV file
csv_file_path = 'intents.csv'
df.to_csv(csv_file_path, index=False)

print(f"CSV file saved: {csv_file_path}")
