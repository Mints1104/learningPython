import numpy as np
import pandas as pd
import os
import json 
import requests
from collections import defaultdict, Counter, deque
import heapq
from ListNode import ListNode

from Car import Car
name = "Harun"
score = 100
if(score > 50):
    print("Based G you passed")

for i in range(5):
    print(i)
    
    
scores = [10, 20, 30, 40, 50]
for score in scores:
    print(score)




def two_sum(nums,target):
    seen_numbers = {}
    
    for index, num in enumerate(nums):
        complement = target - num
        
        if complement in seen_numbers:
            return [seen_numbers[complement],index]
        seen_numbers[num] = index 
    return []

def palindrome(s):
    left = 0
    right = len(s) - 1
    
    while left < right:
        
        while left < right and not s[left].isalnum():
            left += 1
        while left < right and not s[right].isalnum():
            right -=1
        
        if s[left].lower() !=  s[right].lower():
            return False
        
        left += 1
        right -= 1
    return True



squares = [x * x for x in range(10)]
even_squares = [x * x for x in range(10) if x % 2 == 0]

my_array = np.array([1,2,3,4])
doubled_array = my_array * 2 

data = {'Name': ['Alice', 'Bob'], 'Age': [30, 25]}
df = pd.DataFrame(data)

print(df)
greaterThanTwo = [1,2,3,4]
greaterThanTwo = [x  for x in greaterThanTwo if x > 2]
print(greaterThanTwo)

def validAnagram(s1,s2):
    dictionary = {}
    if len(s1) != len(s2):
        return False
    for char in s1:
        dictionary[char] = dictionary.get(char,0) + 1
    for char in s2:
        if char not in dictionary or dictionary[char] == 0:
            return False
        dictionary[char] -=1
    
    return True




data = pd.read_csv("sales.csv")
df = pd.DataFrame(data)

data['Revenue'] = data['Price'] * data['QuantitySold']
total_revenue = sum(data['Revenue'])
print(total_revenue)
sorted_df = df.sort_values(by='Price', ascending=False)
most_expensive_product = sorted_df['Product'].iloc[0]
print(most_expensive_product)

sorted_df_electronics = df[df['Category'] == 'Electronics']
print(sorted_df_electronics)
total_electronics_sold = sum(sorted_df_electronics['QuantitySold'])
print(total_electronics_sold)

print(f"Total Revenue: ${total_revenue}")
print(f"Most Expensive Product: {most_expensive_product}")
print(f"Total Electronics Sold: {total_electronics_sold}")

def contains_duplicate(nums):
    seen_numbers = {}
    
    for num in nums:
        if num  in seen_numbers:
            return True
        seen_numbers.add(num)
    return False
            
def fizzbuzz():
    
    for n in range(101):
        if (n % 5 == 0) and  (n % 3 == 0):
            print("FizzBuzz")  
        elif n % 3 == 0:
            print("Fizz") 
        else:
            print("Buzz")

def reverse_string(s):
    left = 0
    right = len(s) - 1
    
    while(left < right):
        s[left], s[right] = s[right], s[left]
        left +=1
        right -=1
    return s 

my_list = ["h", "e", "l", "l", "o"]
reverse_string(my_list)
print(my_list)

def roman_to_int(s):
    roman_map = {
        'I': 1,
        'V': 5,
        'X': 10,
        'L': 50,
        'C': 100,
        'D': 500,
        'M': 1000
    }
    
    total = 0
    for i in range(len(s) -1):
        current_val = roman_map[s[i]]
        next_val = roman_map[s[i+1]]
        
        if current_val < next_val:
            total -= current_val
        else:
            total += current_val
    total += roman_map[s[-1]]
    return total
  

print(roman_to_int("MCMXCIV"))


def majority_element(nums):
    dict = {}
    size = len(nums)
    print(size)
    
    for n in nums:
        dict[n] = dict.get(n,0) + 1
            
    for num,freq in dict.items():
        if freq > size / 2:
            return num
    return None

print(majority_element([1,1,2,3,4,5,5,5,6,7,8,9,10]))

try:
    with open('example.txt','r') as file:
        for line in file:
        # .strip() removes any leading/trailing whitespace, including the newline character
            print(line.strip())
except FileNotFoundError:
    print("File not found.")
    
lines_to_write = ["First line.", "Second line.", "Third line."]
with open('output.txt','w') as file:
    for line in lines_to_write:
        file.write(line + '\n')
        
output = []
try:
    with open('input.txt','r') as file:
        
        for line in file:
            
          output.append(line.upper())
except FileNotFoundError:
    print("File not found")


with open('output.txt','w') as file:
    for line in output:
        file.write(line)
        
try:
    with open('input.txt','r') as infile, open('output.txt','w') as outfile:
        for line in infile:
            outfile.write(line.upper())
except FileNotFoundError:
    print("The input file was not found")
    
def random(word1, word2):
    size = max(len(word1), len(word2))
    
    chars = []
    chars.append(word1[0])  # append first char of word1
    s = "".join(chars)       # convert list to string
    return s

def mergeAlternately(word1, word2):
    size = max(len(word1), len(word2))
    counter = 0
    chars = []
    print((f"Word 1 Length: {len(word1)}"))
    print((f"Word 2 Length: {len(word2)}"))

    for n in range(size):
        print(f"Counter: {counter}")

        if counter >= len(word1):
            chars.append(word2[n])
        elif counter >= len(word2):
            chars.append(word1[n])
        else:
            chars.append(word1[n])
            chars.append(word2[n])
            counter+=1
    
    
    
    s = "".join(chars)      
    return s

print(mergeAlternately("ab","pqrs"))

        
json_string = '{"name": "Harun","score": 99, "is_active": true}'      
user_data = json.loads(json_string)

print(user_data["name"])
print(user_data["score"])

user_dict = {
    "name": "Alice",
    "level": 15,
    "items": ["sword","shield"]
}
json_string = json.dumps(user_dict,indent=4)

print(json_string)
        
weather_json_string = '{"city": "Bristol", "temperature": 25, "condition": "Sunny"}'
weather_data = json.loads(weather_json_string)
weather_data["temperature"] = 28
weather_data["wind_speed"] = "15km/h"
print(weather_data)
new_weather_json_string = json.dumps(weather_data,indent=4)


def max_profit(prices):
    min_price = float('inf')
    max_profit = 0

   
    for  price in prices:
        
        if price < min_price:
            min_price = price
        elif price - min_price > max_profit:
            max_profit = price - min_price
            
        return max_profit
    
    
    
    
    

print(max_profit([7,1,5,3,6,4]))

def get_random_fact():
    try:
        response = requests.get("https://uselessfacts.jsph.pl/random.json?language=en")
        response.raise_for_status
        data = response.json()
        fact = data['text']
        print("Here's a random fact:")
        print(fact)
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        


d = defaultdict(int)
words = ["apple","banana","apple"]
for word in words:
    d[word] += 1
    
print(d)

more_words = ["apple", "banana", "apple", "orange", "banana", "apple"]
word_counts = Counter(more_words)
print(word_counts)
print(word_counts.most_common(1))

arr = [1,2,3,1,1,2,5,6,7,8]
def majority_element_easy(nums):
    num_count = Counter(nums)
    return num_count.most_common(1)[0][0]

print(majority_element_easy(arr))

def removeDuplicates(nums):
    if not nums:
        return 0
    slow = 0  
    # The fast pointer iterates through the array to find the next unique element.
    for fast in range(1, len(nums)):
        # If we find a new number that is different from our last unique number...
        if nums[fast] != nums[slow]:
            # ...we move the slow pointer forward...
            slow += 1
            # ...and copy the new unique number into that position.
            nums[slow] = nums[fast]
    return slow + 1


def remove_third_dup(nums):
    if len(nums) <= 2:
        return len(nums)
    slow = 2
    for fast in range(2, len(nums)):
        if nums[fast] != nums[slow-2]:
            nums[slow] = nums[fast]
            slow +=1
    return slow
                
def valid_palindrome(s):
    s2 = ''.join([char for char in s if char.isalnum()])
    left = 0
    right = len(s2) - 1
    while left < right:
        
        if s2[left].lower() != s2[right].lower():
            return False
        
        
        
        left +=1
        right -=1
    return True

print(valid_palindrome("race a car"))

def convert_to_binary(n):
    res = ''
    while n > 0:
        res = str(n & 1) + res
        n >>= 1
    return res 

print(convert_to_binary(7))

# --- Python Slicing Examples ---

my_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
my_string = "Hello, World!"

print("--- Basic Slicing ---")
# Get a sublist from index 2 up to (but not including) index 5
# [start:stop]
print(f"my_list[2:5] -> {my_list[2:5]}")  # Output: [2, 3, 4]

# Get everything from the beginning up to index 4
print(f"my_list[:4] -> {my_list[:4]}")    # Output: [0, 1, 2, 3]

# Get everything from index 6 to the end
print(f"my_list[6:] -> {my_list[6:]}")    # Output: [6, 7, 8, 9]

# Get a copy of the entire list
print(f"my_list[:] -> {my_list[:]}")      # Output: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

print("\n--- Slicing with a Step ---")
# Get the whole list, but skip every other element
# [start:stop:step]
print(f"my_list[::2] -> {my_list[::2]}")  # Output: [0, 2, 4, 6, 8]

# Get elements from index 1 to 7, skipping every other element
print(f"my_list[1:8:2] -> {my_list[1:8:2]}") # Output: [1, 3, 5, 7]

print("\n--- Slicing with Negative Indices ---")
# Negative indices count from the end of the list

# Get the last 3 elements
print(f"my_list[-3:] -> {my_list[-3:]}")  # Output: [7, 8, 9]

# Get everything EXCEPT the last 2 elements
print(f"my_list[:-2] -> {my_list[:-2]}") # Output: [0, 1, 2, 3, 4, 5, 6, 7]

print("\n--- Common Use Case: Reversing a Sequence ---")
# The step can be negative, which is a clever way to reverse a list or string
print(f"my_list[::-1] -> {my_list[::-1]}") # Output: [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
print(f"my_string[::-1] -> {my_string[::-1]}") # Output: "!dlroW ,olleH"

def find_difference_two_arrays(nums1,nums2):
    temp1 = nums1.copy()
    temp2 = nums2.copy()
    nums1 = [num for num in nums1 if num not in temp2]
    nums2 = [num for num in nums2 if num not in temp1]
    return [nums1,nums2]

test1 = [1,2,3]
test2 = [2,4,6]


print(find_difference_two_arrays(test1,test2))

def unique_number_of_occurences(nums):
    count = {}
    
    for num in nums:
        count[num] = count.get(num,0) + 1
    seen = set()
    for val in count.values:
        if val in seen:
            return False
        seen.add(val)
    return True
        

def climbStairs(n):
    if n<=2:
        return n 
    
    one_step_before = 2
    two_steps_before = 1
    
    for _ in range(3, n+1):
        current_ways = one_step_before + two_steps_before
        two_steps_before = one_step_before
        one_step_before = current_ways
    return one_step_before

def isHappy(n):
    seen_numbers = set()
    
    while n != 1:
        if n in seen_numbers:
            return False 
        #add current number to set
        seen_numbers.add(n)
        #calculate sum of squares of the digits for the next numb
        sum_of_squares = 0
        num_str = str(n)
        for digit_char in num_str:
            digit_int = int(digit_char)
            sum_of_squares += digit_int * digit_int
            
        n = sum_of_squares
    return True

print(isHappy(19))
print(isHappy(2))
        
def factorial(n):
    if n == 0 or n == 1:
        return 1
        
    return n * factorial(n -1)
    
print(f"Factorial: {factorial(5)}")

def top_k_frequent_elements_heap(nums,k):
    counts = Counter(nums)
    min_heap = []
    for elem, count in counts.items():
        heapq.heappush(min_heap,(count,elem))
        
        if len(min_heap) > k:
            heapq.heappop(min_heap)
    
    return [item[1] for item in min_heap]

        
def top_k_frequent_elements(nums,k):
    counts = Counter(nums)
    top_k = counts.most_common(k)
    return [item[0] for item in top_k]

arr = [1,3,4,3,0,4,9,4,1,2,3,3,4,4,5,9,6,7]
print(top_k_frequent_elements(arr,2))

pairs = [(1, 'a'), (2, 'b'), (3, 'c')]
letters = []

for num, letter in pairs:
    letters.append(letter)
    
letters = [letter for num, letter in pairs]

string = "leetcode"
result = string.find("leeto")
print(result)

def find_index_of_first_occurence(haystack, needle):
    n = len(haystack)
    m = len(needle)
    if m == 0:
      return 0
  
    for i in range(n - m + 1):
      current_slice = haystack[i: i + m]
      
      if current_slice == needle:
          return i
      
    return -1
            
print(find_index_of_first_occurence("sadbutsad","sad"))

def isomorphic_strings(s,t):
    
    if len(s) != len(t):
        return False
    map_s_to_t = {}
    map_t_to_s = {}
    for i in range(len(s)):
        s_char = s[i]
        t_char = t[i]
        if s_char in map_s_to_t:
            if map_s_to_t[s_char] != t_char:
                return False
        else:
            if t_char in map_t_to_s:
                return False
            
        map_s_to_t[s_char] = t_char
        map_t_to_s[t_char] = s_char
        
    return True

def hasCycle(self,head):
    
    slow = head
    fast = head
    
    while fast is not None and fast.next is not None:
        slow = self.next
        fast = self.next.next
        if slow == fast:
            return True
        
    return False 

def maxDepth(self, root):
    if root is None:
        return 0
    
    left_depth = self.maxDepth(root.left)
    right_depth = self.maxDepth(root.right)
    return 1 + max(left_depth,right_depth)
    
def minDepth(self,root):
    if root is None:
        return 0
    queue = deque()
    queue.append((root,1))
    
    while queue:
        current_node, current_depth = queue.popleft()
        
        if current_node.left is None and current_node.right is None:
            return current_depth
        
        if current_node.left:
            queue.append((current_node.left, current_depth +1))
        if current_node.right:
            queue.append((current_node.right, current_depth + 1))
    
            
   
iso_s = "egg"
iso_t = "add"
print(isomorphic_strings(iso_s,iso_t))    

