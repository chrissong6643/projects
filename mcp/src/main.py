def decorator(func):
    def inner(*args,**kwargs):
        print("before")
        print(func(*args, **kwargs))
        return inner
    return decorator



@decorator
def a():
    print("a")

a()