from Contact import Contact 
import json

class ContactBook:
    def __init__(self):
        self.contacts = []
        
    def add_contact(self,contact):
        self.contacts.append(contact)
        print(f"Contact for {contact.name} added.")
    
    def view_contacts(self):
        if not self.contacts:
            print("Your contact book is empty")
            return
        print("\n-- Your Contacts ---")
        for i, contact in enumerate(self.contacts, start = 1):
            print(f"{i}. Name: {contact.name}")
            print(f"   Phone: {contact.phone_number}")
            print(f"   Email: {contact.email}")
            print("--------------------------")
    def search_contact(self,name):
        search_name  = input("Enter the name to search for: ")
        found_contact = False
        
        for contact in self.contacts:
            if contact.name.lower() == search_name.lower():
                print("\n--- Contact Found ---")
                print(f"Name: {contact.name}")
                print(f"Phone Number: {contact.phone_number}")
                print(f"Email: {contact.email}")
                found_contact = True
                break
        if not found_contact:
            print(f"Contact '{search_name}' not found.")
    
    def save_contact(self,filename="contacts.json"):
        contact_dict = []
        for contact in self.contacts:
            contact_dict.append({
                "name": contact.name,
                "phone_number": contact.phone_number,
                "email": contact.email
            })
        with open(filename, 'w') as file:
            json.dump(contact_dict, file, indent=4)
        print("Contacts saved successfully.")
    def load_contacts(self, filename="contacts.json"):
        try:
            with open(filename, 'r') as file:
                contacts = json.load(file)
                
                for contact in contacts:
                    contact = Contact(contact["name"], contact["phone_number"],contact["email"])
                    self.contacts.append(contact)
                    print("Contacts list loaded") 
            
            
            
            
        except FileNotFoundError:
            print(f"File {filename} was not found.")
            
                
def main():
    book = ContactBook()

    while True:
        print("\n--- Contact Book Menu ---")
        print("1. Add Contact")
        print("2. View All Contacts")
        print("3. Search for a Contact")
        print("4. Save contact list")
        print("5. Load contact list")
        print("6. Exit")
        
        choice = input("Enter your choice (1-4): ")

        if choice == '1':
            name = input("Enter name: ")
            phone = input("Enter phone number: ")
            email = input("Enter email: ")
            new_contact = Contact(name, phone, email)
            book.add_contact(new_contact)
        elif choice == '2':
            book.view_contacts()
        elif choice == '3':
            book.search_contact()
        elif choice == '4':
            book.save_contact()
        elif choice =='5':
            book.load_contacts()
        elif choice == '6':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 4.")
            
            
main()