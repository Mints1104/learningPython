from SavingsAccount import SavingsAccount
class Customer():
    
    def __init__(self, name, account_number, interest_rate):
        self.name = name
        self.account = SavingsAccount(account_number,interest_rate)