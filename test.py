def is_prime(number):
    if number < 2:
        return False
    for i in range(2,int(number)):
        if number % i == 0:
            return False
    return True
n = input()
if(is_prime(int(n))):
    print(f'{n} is prime')
else :
    print(f'{n} not prime')