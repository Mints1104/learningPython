from Account import Account
class SavingsAccount(Account):
    def __init__(self, account_number, interest_rate, balance=0):
        super().__init__(account_number, balance)
        self.interest_rate = interest_rate
        
    def add_interest(self):
        interest = self.balance * self.interest_rate
        self.balance += interest