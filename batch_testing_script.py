#Batch testing information 

def report_tests():
    
    tests = {}
    
    tests[1] = {'ENCODER'    : 'resnet34' , 
                'train_size' : 75         ,
                'batch_size' : 16         , 
                'EPOCHS'     : 40         , 
                'lr'         : 0.0001     ,
                'aug_angle'  : 25         ,
                'aug_scale'  : [1,1.5]    , 
                'flip_prob'  : 0.5        ,
                'num_workers': 1          , 
                'dropout'    : 0.5   
                 }
    
    tests[2] = {'ENCODER'    : 'resnet34' , 
                'train_size' : 75         ,
                'batch_size' : 16         , 
                'EPOCHS'     : 100        , 
                'lr'         : 0.0001     ,
                'aug_angle'  : 25         ,
                'aug_scale'  : [1,1.5]    , 
                'flip_prob'  : 0.5        ,
                'num_workers': 1          ,
                'dropout'    : 0.5       
               }
    
    return tests