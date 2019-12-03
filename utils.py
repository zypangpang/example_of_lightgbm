def debug_wrapper(func):
    def innerFunc(*args,**kwargs):
        print("********************** BEGIN **********************")
        func(*args,**kwargs)
        print("********************** END **********************")
    return innerFunc
