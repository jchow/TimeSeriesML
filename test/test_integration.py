import logging
from unittest import TestCase

from fundamentalmodeldatapreparer import FundamentalModelDataPreparer
from fundamentalworker import FundamentalWorker


class TestIntegration(TestCase):

    def test_end2end(self):
        '''
        preparer = FundamentalModelDataPreparer()
        worker = FundamentalWorker()
        tickers = ['MSFT', 'AAPL', 'CSCO', 'IBM', 'UTX', 'V']
        dataset, labels = preparer.get_dataset_for_RNN(tickers)
        model, test_set, scaler = worker.build_save_model_LSTM(dataset, labels)
        result, mse = worker.predict(test_set, scaler)

        # Plot graphs
        print('---- data labels ----')
        print(labels)
        print('==== result ====')
        print('predicted ----')
        print(result[0])
        print('validate test ----')
        print(result[1])
        y_predicted = result[0]
        y_test = result[1]

        plt.figure(figsize=(5.5, 5.5))
        plt.plot(range(len(y_predicted)), y_predicted, linestyle='-', marker='*', color='r')
        plt.plot(range(len(y_test)), y_test, linestyle='-', marker='.', color='b')
        plt.legend(['Actual', 'Predicted'], loc=2)
        plt.show()
        '''

    def test_LSTM(self):
        preparer = FundamentalModelDataPreparer(file='/tmp/test_prepare.log', loglevel=logging.DEBUG)
        worker = FundamentalWorker(file='/tmp/test_worker_lstm.log', loglevel=logging.DEBUG)
        data_array, labels = preparer.get_data_from_intrinio_file()

        save_weights_at, test_set = worker.build_save_model_LSTM(data_array, labels, 'intrinio')
        y, rmse = worker.predict(save_weights_at, test_set)

        print('==== result LSTM ====')
        print('predicted y = ', y)
        print('rmse = ', rmse)
'''
    def test_baseline(self):
        preparer = FundamentalModelDataPreparer(file='/tmp/test_prepare.log', loglevel=logging.DEBUG)
        data_array, labels = preparer.get_data_from_intrinio_file()

        worker = FundamentalWorker(file='/tmp/test_worker_randomforest.log', loglevel=logging.DEBUG)
        y, rmse = worker.predict_baseline(data_array, labels)

        print('==== result base line ====')
        print('predicted y = ', y)
        print('rmse = ', rmse)

    def test_RandomForest(self):
        preparer = FundamentalModelDataPreparer(file='/tmp/test_prepare.log', loglevel=logging.DEBUG)
        data_array, labels = preparer.get_data_from_intrinio_file()

        worker = FundamentalWorker(file='/tmp/test_worker_randomforest.log', loglevel=logging.DEBUG)
        y, rmse = worker.predict_random_forest(data_array, labels)

        print('==== result random forest ====')
        print('predicted y = ', y)
        print('rmse = ', rmse)

    def test_light_gbm(self):
        preparer = FundamentalModelDataPreparer(file='/tmp/test_prepare.log', loglevel=logging.DEBUG)
        data_array, labels = preparer.get_data_from_intrinio_file()

        worker = FundamentalWorker(file='/tmp/test_worker_randomforest.log', loglevel=logging.DEBUG)
        y, rmse = worker.predict_light_gbm(data_array, labels)

        print('==== result light gbm ====')
        print('predicted y = ', y)
        print('rmse = ', rmse)
'''

'''
---- with 0.95 threshold when cleaning data ----

I0707 14:54:49.554071 139738452940544 fundamentalmodeldatapreparer.py:102] final data shape (291, 64)
I0707 14:54:49.567022 139738452940544 fundamentalworker.py:24] creating an instance of data preparer
I0707 14:54:49.574747 139738452940544 fundamentalworker.py:105] -- Split data -- 
==== result random forest ====
('predicted y = ', array([0.35534381, 0.40724636, 0.538027  , 0.41254207, 0.39099143,
       0.41078843, 0.31797703, 0.33390266, 0.36161684, 0.40014162,
       0.35064625, 0.33974216, 0.41375088, 0.36538483, 0.36163735,
       0.34545069, 0.41312047, 0.32773141, 0.42075741, 0.36014919,
       0.38112147, 0.49918555, 0.39110522, 0.31795373, 0.38464597,
       0.37650844, 0.3759219 , 0.40916385, 0.32423608, 0.38981992,
       0.37671358, 0.40641734, 0.36161672, 0.44404183, 0.34491164,
       0.46289183, 0.3763526 , 0.42018315, 0.35150486, 0.37356765,
       0.33079247, 0.35701569, 0.34791967, 0.35162783, 0.32991951,
       0.39677881, 0.35482518, 0.344449  , 0.49048745, 0.32705952,
       0.33302896, 0.41032713, 0.32703991, 0.39191037, 0.36896944,
       0.4069203 , 0.3755241 , 0.44141924, 0.37454849]))
('rmse = ', 0.1236711744275953)

I0707 14:57:17.778769 139738452940544 fundamentalmodeldatapreparer.py:102] final data shape (291, 64)
I0707 14:57:17.799454 139738452940544 fundamentalworker.py:24] creating an instance of data preparer
I0707 14:57:17.800698 139738452940544 fundamentalworker.py:105] -- Split data -- 
==== result base line ====
('predicted y = ', array([0.35519628, 0.35322834, 0.51407134, 0.17439339, 0.4233187 ,
       0.23961284, 0.34096639, 0.3312929 , 0.22554149, 0.49748444,
       0.51010659, 0.39758949, 0.48992956, 0.49698274, 0.27844327,
       0.56941944, 0.22563137, 0.36432147, 0.30725937, 0.30352297,
       0.43671096, 0.14809342, 0.25947515, 0.39598187, 0.48410318,
       0.41592805, 0.411312  , 0.59009844, 0.64266907, 0.66626236,
       0.53825017, 0.39992765, 0.3278946 , 0.50201487, 0.25147897,
       0.55919161, 0.43793155, 0.39011337, 0.18666945, 0.39355642,
       0.49195733, 0.3343958 , 0.45039417, 0.38695064, 0.42188851,
       0.39642911, 0.43729271, 0.28453126, 0.42741071, 0.50258412,
       0.2653744 , 0.47636561, 0.45738406, 0.55109895, 0.43137864,
       0.5246248 , 0.30231211, 0.44746221]))
('rmse = ', 0.1578942357728254)

I0707 14:57:24.641089 139738452940544 fundamentalmodeldatapreparer.py:102] final data shape (291, 64)
I0707 14:57:24.655613 139738452940544 fundamentalworker.py:24] creating an instance of data preparer
I0707 14:57:24.656894 139738452940544 fundamentalworker.py:105] -- Split data -- 
==== result light gbm ====
('predicted y = ', array([0.33098512, 0.35222371, 0.53617001, 0.40357449, 0.48774164,
       0.41963897, 0.25201304, 0.34613686, 0.34578819, 0.3598679 ,
       0.27723048, 0.31882286, 0.41370724, 0.37467726, 0.30945084,
       0.40669754, 0.45852205, 0.35396632, 0.37004502, 0.3708688 ,
       0.28704021, 0.48952427, 0.3954143 , 0.28008889, 0.36779503,
       0.32057532, 0.38145682, 0.49751851, 0.29870021, 0.42464319,
       0.37015384, 0.47103115, 0.34348585, 0.44450781, 0.2917742 ,
       0.42733724, 0.30878495, 0.49666305, 0.3362824 , 0.40669382,
       0.32364166, 0.38030374, 0.30986082, 0.28561622, 0.38511962,
       0.41296032, 0.26713096, 0.30827136, 0.45093962, 0.34186285,
       0.31579568, 0.45670524, 0.33834335, 0.44715654, 0.39062212,
       0.37275936, 0.37426899, 0.44316008, 0.37282489]))
('rmse = ', 0.1280062937791651)



---- with 0.99 threshold when cleaning data ----

I0707 14:41:28.463784 140345204639488 fundamentalmodeldatapreparer.py:102] final data shape (307, 54)
I0707 14:41:28.470638 140345204639488 fundamentalworker.py:24] creating an instance of data preparer
I0707 14:41:28.475944 140345204639488 fundamentalworker.py:105] -- Split data -- 
==== result base line ====
('predicted y = ', array([0.17439339, 0.34096639, 0.23961284, 0.22554149, 0.4233187 ,
       0.35519628, 0.35322834, 0.3312929 , 0.49748444, 0.39758949,
       0.51010659, 0.19469023, 0.53171221, 0.56941944, 0.27844327,
       0.48992956, 0.49698274, 0.43671096, 0.39598187, 0.25947515,
       0.30725937, 0.27400054, 0.22563137, 0.28482106, 0.30352297,
       0.36432147, 0.66626236, 0.59009844, 0.41592805, 0.3278946 ,
       0.64266907, 0.411312  , 0.53825017, 0.39992765, 0.50201487,
       0.39355642, 0.3343958 , 0.43793155, 0.39011337, 0.25147897,
       0.55919161, 0.45039417, 0.18666945, 0.49195733, 0.42188851,
       0.38695064, 0.2653744 , 0.39642911, 0.42741071, 0.43729271,
       0.50258412, 0.47636561, 0.28453126, 0.43137864, 0.55109895,
       0.5246248 , 0.60856222, 0.30231211, 0.1839078 , 0.45738406,
       0.44746221]))
('rmse = ', 0.1553597434050116)

I0707 14:40:04.288475 140345204639488 fundamentalmodeldatapreparer.py:102] final data shape (307, 54)
I0707 14:40:04.295069 140345204639488 fundamentalworker.py:24] creating an instance of data preparer
I0707 14:40:04.296380 140345204639488 fundamentalworker.py:105] -- Split data -- 
==== result random forest ====
('predicted y = ', array([0.39244395, 0.30408578, 0.38282393, 0.35675538, 0.35364764,
       0.32532338, 0.40291782, 0.35537162, 0.39916249, 0.30769166,
       0.38016661, 0.36675316, 0.37380739, 0.3265626 , 0.35634712,
       0.4166954 , 0.35918837, 0.38606982, 0.30757995, 0.38844979,
       0.4236076 , 0.52129991, 0.38775055, 0.37325849, 0.33256365,
       0.29479886, 0.37909798, 0.40375531, 0.40796472, 0.34971087,
       0.31539549, 0.3879706 , 0.37553945, 0.38129219, 0.3945611 ,
       0.37227839, 0.30460824, 0.38312765, 0.46537123, 0.31865058,
       0.40576724, 0.32283441, 0.36759933, 0.36693274, 0.29655015,
       0.37491391, 0.33818657, 0.3899616 , 0.48915546, 0.35373806,
       0.3183402 , 0.39533001, 0.36107857, 0.32307285, 0.37481491,
       0.39946328, 0.32692916, 0.37453167, 0.39764684, 0.28159122,
       0.40667014, 0.34717893]))
('rmse = ', 0.13234630404795433)


I0707 14:41:36.510744 140345204639488 fundamentalmodeldatapreparer.py:102] final data shape (307, 54)
I0707 14:41:36.519081 140345204639488 fundamentalworker.py:24] creating an instance of data preparer
I0707 14:41:36.522558 140345204639488 fundamentalworker.py:105] -- Split data -- 
==== result light gbm ====
('predicted y = ', array([0.36970006, 0.26864795, 0.40472391, 0.30358362, 0.43780825,
       0.29375074, 0.46916666, 0.3364034 , 0.42743501, 0.27378074,
       0.25581496, 0.34863993, 0.36272681, 0.37349877, 0.34027015,
       0.33329741, 0.38916667, 0.36376548, 0.29555132, 0.3693812 ,
       0.272879  , 0.5094816 , 0.34831977, 0.34785719, 0.27369764,
       0.30113309, 0.43627372, 0.48056574, 0.24926034, 0.37377108,
       0.25860279, 0.35142478, 0.38261427, 0.45078491, 0.23565602,
       0.36453993, 0.27478562, 0.3701224 , 0.38781359, 0.27567277,
       0.32548885, 0.30006995, 0.41311845, 0.31924072, 0.35094038,
       0.27886246, 0.28336593, 0.41215664, 0.37231115, 0.37878675,
       0.37319836, 0.3092363 , 0.37702224, 0.27752202, 0.40170253,
       0.37781466, 0.32219228, 0.34879696, 0.29300918, 0.29368592,
       0.3502555 , 0.34555593]))
('rmse = ', 0.1397422359287333)

        '''