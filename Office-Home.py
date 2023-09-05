from data import *
from utilities import *
from networks import *
# import matplotlib.pyplot as plt
import numpy as np

def skip(data, label, is_train):
    return False
batch_size = 16 


import sys



domain_train = sys.argv[1]
domain_test = sys.argv[2]
setGPU(sys.argv[3])
store_name = sys.argv[4]
loss_MSE_value = float(sys.argv[5])
three_domain_loss_value = float(sys.argv[6])



log = Logger(store_name + '/step_2', clear=True)
print('domain_train', domain_train)
print('domain_test', domain_test)
# Art  1089  
# Clipart  1675
# Product  1785
# RealWorld  1811


def transform(data, label, is_train):
    label = one_hot(26, label)
    data = tl.prepro.crop(data, 224, 224, is_random=is_train)
    data = np.transpose(data, [2, 0, 1])
    data = np.asarray(data, np.float32) / 255.0
    return data, label
ds = FileListDataset('/home/username/data/OfficeHomeDataset_10072016/'+domain_train + '_shared_list.txt', \
    '/home/username/data/office/', transform=transform, skip_pred=skip, is_train=True, imsize=256)
source_train = CustomDataLoader(ds, batch_size=batch_size, num_threads=2)

def transform(data, label, is_train):
    if label in range(25):
        label = one_hot(26, label)
    else:
        label = one_hot(26,25)
    data = tl.prepro.crop(data, 224, 224, is_random=is_train)
    data = np.transpose(data, [2, 0, 1])
    data = np.asarray(data, np.float32) / 255.0
    return data, label
ds1 = FileListDataset('/home/username/data/OfficeHomeDataset_10072016/'+ domain_test+ '_list.txt', \
    '/home/username/data/office/', transform=transform, skip_pred=skip, is_train=True, imsize=256)
target_train = CustomDataLoader(ds1, batch_size=batch_size, num_threads=2)

def transform(data, label, is_train):
    label = one_hot(65,label)
    data = tl.prepro.crop(data, 224, 224, is_random=is_train)
    data = np.transpose(data, [2, 0, 1])
    data = np.asarray(data, np.float32) / 255.0
    return data, label
ds2 =FileListDataset('/home/username/data/OfficeHomeDataset_10072016/'+ domain_test+ '_list.txt', \
    '/home/username/data/office/', transform=transform, skip_pred=skip, is_train=False, imsize=256)
target_test = CustomDataLoader(ds2, batch_size=batch_size, num_threads=2)


discriminator_p = Discriminator(n = 25).cuda()  # 10 binary classifier
discriminator = LargeAdversarialNetwork(256).cuda()

feature_extractor_fix = ResNetFc(model_name='resnet50',model_path='/home/username/data/pytorchModels/resnet50.pth')
feature_extractor_nofix = ResNetFc(model_name='resnet50',model_path='/home/username/data/pytorchModels/resnet50.pth')

cls_upper = CLS(feature_extractor_fix.output_num(), 26, bottle_neck_dim=256)
cls_down = CLS(feature_extractor_nofix.output_num(), 26, bottle_neck_dim=256)

net_upper = nn.Sequential(feature_extractor_fix, cls_upper).cuda()
net_down = nn.Sequential(feature_extractor_nofix, cls_down).cuda()


three_domain_discriminator = Discriminator2(n = 3).cuda() 



scheduler = lambda step, initial_lr : inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=10000)

optimizer_discriminator = OptimWithSheduler(optim.SGD(discriminator.parameters(), lr=5e-4, weight_decay=5e-4, momentum=0.9, nesterov=True),
                            scheduler)


optimizer_feature_extractor_fix = OptimWithSheduler(optim.SGD(feature_extractor_fix.parameters(), lr=5e-5, weight_decay=5e-4, momentum=0.9, nesterov=True),
                            scheduler)

optimizer_feature_extractor_nofix = OptimWithSheduler(optim.SGD(feature_extractor_nofix.parameters(), lr=5e-5, weight_decay=5e-4, momentum=0.9, nesterov=True),
                            scheduler)

optimizer_cls_upper = OptimWithSheduler(optim.SGD(cls_upper.parameters(), lr=5e-4, weight_decay=5e-4, momentum=0.9, nesterov=True),
                            scheduler)

optimizer_cls_down = OptimWithSheduler(optim.SGD(cls_down.parameters(), lr=5e-4, weight_decay=5e-4, momentum=0.9, nesterov=True),
                            scheduler)

optimizer_discriminator_p = OptimWithSheduler(optim.SGD(discriminator_p.parameters(), lr=1e-3, weight_decay=5e-4, momentum=0.9, nesterov=True),
                            scheduler)

optimizer_three_domain_discriminator = OptimWithSheduler(optim.SGD(three_domain_discriminator.parameters(), lr=5e-4, weight_decay=5e-4, momentum=0.9, nesterov=True),
                            scheduler)


KL = nn.KLDivLoss()
MSE = nn.MSELoss()

# =========================weighted adaptation of the source and target domains      
                    
k=0
while k <6000:
    for (i, ((im_source, label_source), (im_target, label_target))) in enumerate(
            zip(source_train.generator(), target_train.generator())):
        
        im_source = Variable(torch.from_numpy(im_source)).cuda()
        label_source = Variable(torch.from_numpy(label_source)).cuda()
        im_target = Variable(torch.from_numpy(im_target)).cuda()
         
        fs1, feature_source, __, predict_prob_source_upper = net_upper.forward(im_source)
        ft1, feature_target, __, predict_prob_target = net_upper.forward(im_target)


        p0_upper = discriminator_p.forward(fs1)
        p1_upper = discriminator_p.forward(ft1)
        
        p2_upper = torch.sum(p1_upper, dim = -1).detach()
        p3_upper = torch.sum(p0_upper, dim = -1).detach()

        d1_upper = BCELossForMultiClassification(label_source[:,0:25],p0_upper)

        ce_upper = CrossEntropyLoss(label_source, predict_prob_source_upper)



        fs1, feature_source, __, predict_prob_source_down = net_down.forward(im_source)
        ft1, feature_target, __, predict_prob_target = net_down.forward(im_target)

        domain_prob_discriminator_1_source = discriminator.forward(feature_source)
        domain_prob_discriminator_1_target = discriminator.forward(feature_target)
    
        p0_down = discriminator_p.forward(fs1)
        p1_down = discriminator_p.forward(ft1)




        r_unk = torch.sort(p2_upper,dim = 0)[1][:2]  # 001
        feature_otherep = torch.index_select(ft1, 0, r_unk.view(2)) 
        feature_unk, feature_target_unkonwn, _, predict_prob_otherep = cls_down.forward(feature_otherep)


        r = torch.sort(p2_upper,dim = 0)[1][-2:]  # 010
        feature_otherep = torch.index_select(ft1, 0, r.view(2)) 
        _, feature_target_konwn, _, _ = cls_down.forward(feature_otherep)

        r = torch.sort(p3_upper,dim = 0)[1][-2:]  # 100
        feature_otherep = torch.index_select(fs1, 0, r.view(2)) 
        _, feature_source_konwn, _, _ = cls_down.forward(feature_otherep)  



  
        feature_target_unkonwn_labels = Variable(torch.from_numpy(np.concatenate((np.zeros((2,2)), np.ones((2,1))), axis = -1).astype('float32'))).cuda() #001
        feature_target_konwn_labels   = Variable(torch.from_numpy(np.concatenate((np.ones((2,2)),  np.zeros((2,1))), axis = -1).astype('float32'))).cuda()#110
        feature_source_unkonwn_labels = Variable(torch.from_numpy(np.concatenate((np.ones((2,2)),  np.zeros((2,1))), axis = -1).astype('float32'))).cuda()#110

                
        three_domain_discriminator_feature = torch.cat([feature_target_unkonwn,feature_target_konwn, feature_source_konwn], 0)
        three_domain_discriminator_labels  = torch.cat([feature_target_unkonwn_labels,feature_target_konwn_labels, feature_source_unkonwn_labels], 0)

        p0 = three_domain_discriminator.forward(three_domain_discriminator_feature)
        three_domain_loss = BCELossForMultiClassification(three_domain_discriminator_labels,p0)



        ce_ep = CrossEntropyLoss(Variable(torch.from_numpy(np.concatenate((np.zeros((2,25)), np.ones((2,1))), axis = -1).astype('float32'))).cuda(), \
                                    predict_prob_otherep)


        ce_down = CrossEntropyLoss(label_source, predict_prob_source_down)






        loss_MSE_upper =  MSE(p0_upper,p0_down.detach())
        loss_MSE_upper += MSE(p1_upper,p1_down.detach())



        entropy  = EntropyLoss(predict_prob_target, instance_level_weight= p2_upper.contiguous())
        adv_loss = BCELossForMultiClassification(label=torch.ones_like(domain_prob_discriminator_1_source), \
                                                 predict_prob=domain_prob_discriminator_1_source)
        adv_loss += BCELossForMultiClassification(label=torch.ones_like(domain_prob_discriminator_1_target), \
                                                  predict_prob=1 - domain_prob_discriminator_1_target, 
                                                  instance_level_weight = p2_upper.contiguous())


        with OptimizerManager([optimizer_cls_upper, optimizer_discriminator_p,optimizer_feature_extractor_fix]):
            loss = loss_MSE_value * loss_MSE_upper + d1_upper + ce_upper 
            loss.backward()


#---------------------------------------------------------------------------------------------------------------------------
        fs1, feature_source, __, predict_prob_source_upper = net_upper.forward(im_source)
        ft1, feature_target, __, predict_prob_target = net_upper.forward(im_target)


        p0_upper = discriminator_p.forward(fs1)
        p1_upper = discriminator_p.forward(ft1)
        
        p2_upper = torch.sum(p1_upper, dim = -1).detach()
        p3_upper = torch.sum(p0_upper, dim = -1).detach()

        d1_upper = BCELossForMultiClassification(label_source[:,0:25],p0_upper)

        ce_upper = CrossEntropyLoss(label_source, predict_prob_source_upper)



        fs1, feature_source, __, predict_prob_source_down = net_down.forward(im_source)
        ft1, feature_target, __, predict_prob_target = net_down.forward(im_target)

        domain_prob_discriminator_1_source = discriminator.forward(feature_source)
        domain_prob_discriminator_1_target = discriminator.forward(feature_target)
    
        p0_down = discriminator_p.forward(fs1)
        p1_down = discriminator_p.forward(ft1)




        r_unk = torch.sort(p2_upper,dim = 0)[1][:2]  # 001
        feature_otherep = torch.index_select(ft1, 0, r_unk.view(2)) 
        feature_unk, feature_target_unkonwn, _, predict_prob_otherep = cls_down.forward(feature_otherep)


        r = torch.sort(p2_upper,dim = 0)[1][-2:]  # 010
        feature_otherep = torch.index_select(ft1, 0, r.view(2)) 
        _, feature_target_konwn, _, _ = cls_down.forward(feature_otherep)

        r = torch.sort(p3_upper,dim = 0)[1][-2:]  # 100
        feature_otherep = torch.index_select(fs1, 0, r.view(2)) 
        _, feature_source_konwn, _, _ = cls_down.forward(feature_otherep)  



  
        feature_target_unkonwn_labels = Variable(torch.from_numpy(np.concatenate((np.zeros((2,2)), np.ones((2,1))), axis = -1).astype('float32'))).cuda() #001
        feature_target_konwn_labels   = Variable(torch.from_numpy(np.concatenate((np.ones((2,2)),  np.zeros((2,1))), axis = -1).astype('float32'))).cuda()#110
        feature_source_unkonwn_labels = Variable(torch.from_numpy(np.concatenate((np.ones((2,2)),  np.zeros((2,1))), axis = -1).astype('float32'))).cuda()#110

                
        three_domain_discriminator_feature = torch.cat([feature_target_unkonwn,feature_target_konwn, feature_source_konwn], 0)
        three_domain_discriminator_labels  = torch.cat([feature_target_unkonwn_labels,feature_target_konwn_labels, feature_source_unkonwn_labels], 0)

        p0 = three_domain_discriminator.forward(three_domain_discriminator_feature)
        three_domain_loss = BCELossForMultiClassification(three_domain_discriminator_labels,p0)



        ce_ep = CrossEntropyLoss(Variable(torch.from_numpy(np.concatenate((np.zeros((2,25)), np.ones((2,1))), axis = -1).astype('float32'))).cuda(), \
                                    predict_prob_otherep)


        ce_down = CrossEntropyLoss(label_source, predict_prob_source_down)




        loss_MSE_down =  MSE(p0_down,p0_upper.detach())
        loss_MSE_down += MSE(p1_down,p1_upper.detach())


        
        entropy  = EntropyLoss(predict_prob_target, instance_level_weight= p2_upper.contiguous())
        adv_loss = BCELossForMultiClassification(label=torch.ones_like(domain_prob_discriminator_1_source), \
                                                 predict_prob=domain_prob_discriminator_1_source)
        adv_loss += BCELossForMultiClassification(label=torch.ones_like(domain_prob_discriminator_1_target), \
                                                  predict_prob=1 - domain_prob_discriminator_1_target, 
                                                  instance_level_weight = p2_upper.contiguous())


        with OptimizerManager([optimizer_cls_down, optimizer_feature_extractor_nofix,optimizer_discriminator,optimizer_three_domain_discriminator]):
            loss = loss_MSE_value * loss_MSE_down  + ce_down +  0.3 * adv_loss + 0.1 * entropy + three_domain_loss_value * three_domain_loss 

            loss.backward()
            
        k += 1
        log.step += 1

        if log.step % 10 == 1:

            counter_upper = AccuracyCounter()
            counter_upper.addOntBatch(variable_to_numpy(predict_prob_source_upper), variable_to_numpy(label_source))
            acc_train_upper = Variable(torch.from_numpy(np.asarray([counter_upper.reportAccuracy()], dtype=np.float32))).cuda()

            counter_down = AccuracyCounter()
            counter_down.addOntBatch(variable_to_numpy(predict_prob_source_down), variable_to_numpy(label_source))
            acc_train_down = Variable(torch.from_numpy(np.asarray([counter_down.reportAccuracy()], dtype=np.float32))).cuda()
            track_scalars(log, ['ce_down', 'acc_train_down', 'acc_train_upper','adv_loss','entropy',"d1_upper", "loss_MSE_upper","loss_MSE_down","three_domain_loss","ce_ep"], globals())

        if log.step % 100 == 0:
            clear_output()
    print(k)






                 
k=0
while k <3000:
    for (i, ((im_source, label_source), (im_target, label_target))) in enumerate(
            zip(source_train.generator(), target_train.generator())):
        
        im_source = Variable(torch.from_numpy(im_source)).cuda()
        label_source = Variable(torch.from_numpy(label_source)).cuda()
        im_target = Variable(torch.from_numpy(im_target)).cuda()
         
        fs1, feature_source, __, predict_prob_source_upper = net_upper.forward(im_source)
        ft1, feature_target, __, predict_prob_target = net_upper.forward(im_target)


        p0_upper = discriminator_p.forward(fs1)
        p1_upper = discriminator_p.forward(ft1)
        
        p2_upper = torch.sum(p1_upper, dim = -1).detach()
        p3_upper = torch.sum(p0_upper, dim = -1).detach()

        d1_upper = BCELossForMultiClassification(label_source[:,0:25],p0_upper)

        ce_upper = CrossEntropyLoss(label_source, predict_prob_source_upper)



        fs1, feature_source, __, predict_prob_source_down = net_down.forward(im_source)
        ft1, feature_target, __, predict_prob_target = net_down.forward(im_target)

        domain_prob_discriminator_1_source = discriminator.forward(feature_source)
        domain_prob_discriminator_1_target = discriminator.forward(feature_target)
    
        p0_down = discriminator_p.forward(fs1)
        p1_down = discriminator_p.forward(ft1)




        r_unk = torch.sort(p2_upper,dim = 0)[1][:2]  # 001
        feature_otherep = torch.index_select(ft1, 0, r_unk.view(2)) 
        feature_unk, feature_target_unkonwn, _, predict_prob_otherep = cls_down.forward(feature_otherep)


        r = torch.sort(p2_upper,dim = 0)[1][-2:]  # 010
        feature_otherep = torch.index_select(ft1, 0, r.view(2)) 
        _, feature_target_konwn, _, _ = cls_down.forward(feature_otherep)

        r = torch.sort(p3_upper,dim = 0)[1][-2:]  # 100
        feature_otherep = torch.index_select(fs1, 0, r.view(2)) 
        _, feature_source_konwn, _, _ = cls_down.forward(feature_otherep)  



  
        feature_target_unkonwn_labels = Variable(torch.from_numpy(np.concatenate((np.zeros((2,2)), np.ones((2,1))), axis = -1).astype('float32'))).cuda() #001
        feature_target_konwn_labels   = Variable(torch.from_numpy(np.concatenate((np.ones((2,2)),  np.zeros((2,1))), axis = -1).astype('float32'))).cuda()#110
        feature_source_unkonwn_labels = Variable(torch.from_numpy(np.concatenate((np.ones((2,2)),  np.zeros((2,1))), axis = -1).astype('float32'))).cuda()#110

                
        three_domain_discriminator_feature = torch.cat([feature_target_unkonwn,feature_target_konwn, feature_source_konwn], 0)
        three_domain_discriminator_labels  = torch.cat([feature_target_unkonwn_labels,feature_target_konwn_labels, feature_source_unkonwn_labels], 0)

        p0 = three_domain_discriminator.forward(three_domain_discriminator_feature)
        three_domain_loss = BCELossForMultiClassification(three_domain_discriminator_labels,p0)



        ce_ep = CrossEntropyLoss(Variable(torch.from_numpy(np.concatenate((np.zeros((2,25)), np.ones((2,1))), axis = -1).astype('float32'))).cuda(), \
                                    predict_prob_otherep)


        ce_down = CrossEntropyLoss(label_source, predict_prob_source_down)

        loss_MSE_upper =  MSE(p0_upper,p0_down.detach())
        loss_MSE_upper += MSE(p1_upper,p1_down.detach())



        entropy  = EntropyLoss(predict_prob_target, instance_level_weight= p2_upper.contiguous())
        adv_loss = BCELossForMultiClassification(label=torch.ones_like(domain_prob_discriminator_1_source), \
                                                 predict_prob=domain_prob_discriminator_1_source)
        adv_loss += BCELossForMultiClassification(label=torch.ones_like(domain_prob_discriminator_1_target), \
                                                  predict_prob=1 - domain_prob_discriminator_1_target, 
                                                  instance_level_weight = p2_upper.contiguous())


        with OptimizerManager([optimizer_cls_upper, optimizer_discriminator_p,optimizer_feature_extractor_fix]):
            loss = loss_MSE_value * loss_MSE_upper + d1_upper + ce_upper 
            loss.backward()


#---------------------------------------------------------------------------------------------------------------------------
        fs1, feature_source, __, predict_prob_source_upper = net_upper.forward(im_source)
        ft1, feature_target, __, predict_prob_target = net_upper.forward(im_target)


        p0_upper = discriminator_p.forward(fs1)
        p1_upper = discriminator_p.forward(ft1)
        
        p2_upper = torch.sum(p1_upper, dim = -1).detach()
        p3_upper = torch.sum(p0_upper, dim = -1).detach()

        d1_upper = BCELossForMultiClassification(label_source[:,0:25],p0_upper)

        ce_upper = CrossEntropyLoss(label_source, predict_prob_source_upper)



        fs1, feature_source, __, predict_prob_source_down = net_down.forward(im_source)
        ft1, feature_target, __, predict_prob_target = net_down.forward(im_target)

        domain_prob_discriminator_1_source = discriminator.forward(feature_source)
        domain_prob_discriminator_1_target = discriminator.forward(feature_target)
    
        p0_down = discriminator_p.forward(fs1)
        p1_down = discriminator_p.forward(ft1)




        r_unk = torch.sort(p2_upper,dim = 0)[1][:2]  # 001
        feature_otherep = torch.index_select(ft1, 0, r_unk.view(2)) 
        feature_unk, feature_target_unkonwn, _, predict_prob_otherep = cls_down.forward(feature_otherep)


        r = torch.sort(p2_upper,dim = 0)[1][-2:]  # 010
        feature_otherep = torch.index_select(ft1, 0, r.view(2)) 
        _, feature_target_konwn, _, _ = cls_down.forward(feature_otherep)

        r = torch.sort(p3_upper,dim = 0)[1][-2:]  # 100
        feature_otherep = torch.index_select(fs1, 0, r.view(2)) 
        _, feature_source_konwn, _, _ = cls_down.forward(feature_otherep)  



  
        feature_target_unkonwn_labels = Variable(torch.from_numpy(np.concatenate((np.zeros((2,2)), np.ones((2,1))), axis = -1).astype('float32'))).cuda() #001
        feature_target_konwn_labels   = Variable(torch.from_numpy(np.concatenate((np.ones((2,2)),  np.zeros((2,1))), axis = -1).astype('float32'))).cuda()#110
        feature_source_unkonwn_labels = Variable(torch.from_numpy(np.concatenate((np.ones((2,2)),  np.zeros((2,1))), axis = -1).astype('float32'))).cuda()#110

                
        three_domain_discriminator_feature = torch.cat([feature_target_unkonwn,feature_target_konwn, feature_source_konwn], 0)
        three_domain_discriminator_labels  = torch.cat([feature_target_unkonwn_labels,feature_target_konwn_labels, feature_source_unkonwn_labels], 0)

        p0 = three_domain_discriminator.forward(three_domain_discriminator_feature)
        three_domain_loss = BCELossForMultiClassification(three_domain_discriminator_labels,p0)



        ce_ep = CrossEntropyLoss(Variable(torch.from_numpy(np.concatenate((np.zeros((2,25)), np.ones((2,1))), axis = -1).astype('float32'))).cuda(), \
                                    predict_prob_otherep)


        ce_down = CrossEntropyLoss(label_source, predict_prob_source_down)




        loss_MSE_down =  MSE(p0_down,p0_upper.detach())
        loss_MSE_down += MSE(p1_down,p1_upper.detach())


        
        entropy  = EntropyLoss(predict_prob_target, instance_level_weight= p2_upper.contiguous())
        adv_loss = BCELossForMultiClassification(label=torch.ones_like(domain_prob_discriminator_1_source), \
                                                 predict_prob=domain_prob_discriminator_1_source)
        adv_loss += BCELossForMultiClassification(label=torch.ones_like(domain_prob_discriminator_1_target), \
                                                  predict_prob=1 - domain_prob_discriminator_1_target, 
                                                  instance_level_weight = p2_upper.contiguous())


        with OptimizerManager([optimizer_cls_down, optimizer_feature_extractor_nofix,optimizer_discriminator,optimizer_three_domain_discriminator]):
            loss = loss_MSE_value * loss_MSE_down  + ce_down +  0.3 * adv_loss + 0.1 * entropy + three_domain_loss_value * three_domain_loss  + 0.3 * ce_ep

            loss.backward()
            
        k += 1
        log.step += 1

        if log.step % 10 == 1:

            counter_upper = AccuracyCounter()
            counter_upper.addOntBatch(variable_to_numpy(predict_prob_source_upper), variable_to_numpy(label_source))
            acc_train_upper = Variable(torch.from_numpy(np.asarray([counter_upper.reportAccuracy()], dtype=np.float32))).cuda()

            counter_down = AccuracyCounter()
            counter_down.addOntBatch(variable_to_numpy(predict_prob_source_down), variable_to_numpy(label_source))
            acc_train_down = Variable(torch.from_numpy(np.asarray([counter_down.reportAccuracy()], dtype=np.float32))).cuda()
            track_scalars(log, ['ce_down', 'acc_train_down', 'acc_train_upper','adv_loss','entropy',"d1_upper", "loss_MSE_upper","loss_MSE_down","three_domain_loss","ce_ep"], globals())

        if log.step % 100 == 0:
            clear_output()
    print(k)

    with TrainingModeManager([feature_extractor_fix,feature_extractor_nofix, cls_down,cls_upper], train=False) \
                                as mgr, Accumulator(['predict_prob','predict_index', 'label']) as accumulator:
        for (i, (im, label)) in enumerate(target_test.generator()):

            im = Variable(torch.from_numpy(im), volatile=True).cuda()
            label = Variable(torch.from_numpy(label), volatile=True).cuda()


            ft1, feature_target, __, predict_prob = net_down.forward(im)

            p1_upper = discriminator_p.forward(ft1)
            p2_upper = torch.sum(p1_upper, dim = -1).detach()


            predict_prob, label = [variable_to_numpy(x) for x in (predict_prob,label)]
            label = np.argmax(label, axis=-1).reshape(-1, 1)
            predict_index = np.argmax(predict_prob, axis=-1).reshape(-1, 1)
            accumulator.updateData(globals())
            # if i % 10 == 0:
            #     print(i)

    for x in accumulator.keys():
        globals()[x] = accumulator[x]

    y_true = label.flatten()
    y_pred = predict_index.flatten()
    m = extended_confusion_matrix(y_true, y_pred, true_labels=list(range(25))+list(range(25,65)), pred_labels=list(range(26)))


    cm = m
    cm = cm.astype(np.float) / np.sum(cm, axis=1, keepdims=True)
    acc_os_star = sum([cm[i][i] for i in range(25)]) / 25
    acc_os = (acc_os_star * 25 + sum([cm[i][25] for i in range(25, 65)]) / 40) / 26
    #unk = sum([cm[i][25] for i in range(25, 65)]) / 40

    print("acc_os, acc_os_star, unk==",acc_os, acc_os_star, sum([cm[i][25] for i in range(25, 65)]) / 40)




