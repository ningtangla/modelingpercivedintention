import pandas

dataIndex = pd.MultiIndex.from_product([[0, 1], ['x', 'y']], names=['objNum', 'coordinate'])
hypothesisIndex = pd.MultiIndex.from_product([[0, 1],[1, 0], [0], [1.74]], names = ['wolfObjNum', 'sheepObjNum', 'chasingSubtlety', 'escapingPrecision'])
beforeData = pd.DataFrame([[3, 3, 3, 6]], columns = dataIndex)
nowData = pd.DataFrame([[3, 4, 3, 7]], columns = dataIndex)
prior = pd.DataFrame([[0.25, 0.25, 0.25, 0.25]], columns = hypothesisIndex)

__import__('ipdb').set_trace()
