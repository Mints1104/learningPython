class Account:
    def __init__(self, account_number, balance = 0):
        self.account_number = account_number
        self.balance = balance
        
    def deposit(self,amount):
        if amount > 0:
            self.balance += amount
            return True
        return False
        
    def withdraw(self,amount):
        if 0 < amount <= self.balance:
            self.balance -= amount
            return True
        return False
     