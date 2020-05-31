import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QMessageBox
from main_ui import *
from kid_ui import *
from classify import *
import numpy as np
class Myshow(QtWidgets.QWidget, Ui_MainWindow):
    def __init__(self):
        super(Myshow, self).__init__()
        self.setupUi(self)
        self.pushButton_2.clicked.connect(self.ChoosePath)#上传图片
        self.pushButton.clicked.connect(self.Recognition)#开始识别
        self.test_path = './dataset-resized'

    def ChoosePath(self):
        file_name = QtWidgets.QFileDialog.getOpenFileName(self, "open file dialog", self.test_path, "图片(*.jpg)")
        print(file_name[0])
        self.test_path = file_name[0]
        self.lineEdit_2.setText(self.test_path)#显示路径
        self.label_4.setPixmap(QtGui.QPixmap(self.test_path)) #显示待测图片

        # 清空不相关内容
        self.lineEdit.clear()

    def Recognition(self):
        input=self.lineEdit_2.text()#存获取的地址
        garbage=''
        gt=''
        if (input==""):
            print(QMessageBox.warning(self, "警告", "请插入图片", QMessageBox.Yes, QMessageBox.Yes))
            return
        else:
            model = Model()
            model1 = model.load_model()
            print('model loaded')
            # img_path = "./dataset-resized/glass/glass39.jpg"
            img_path = input
            img = image.load_img(img_path, target_size=(300, 300))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            result = model1.predict(img)
            print(generate_result(result))
            result0=generate_result(result)
            garbage = result0
            # 对结果进行分类
            if garbage=="trash":
                gt="干垃圾"
            elif garbage=="paper":
                gt="可回收垃圾"
            elif garbage=="cardboard":
                gt = "可回收垃圾"
            elif garbage=="glass":
                gt = "可回收垃圾"
            elif garbage=="metal":
                gt = "可回收垃圾"
            elif garbage=="plastic":
                gt = "可回收垃圾"
            self.lineEdit.setText(gt)
            print(QMessageBox.information(self, "提醒", "成功识别！该垃圾为：" + garbage, QMessageBox.Yes, QMessageBox.Yes))
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    # 实例化主窗口
    main = QMainWindow()
    main_ui = Ui_MainWindow()
    main_ui.setupUi(main)
    main = Myshow()
    main.setWindowTitle("嘎嘣脆垃圾识别系统")
    # 实例化子窗口
    child = QDialog()
    child.setWindowTitle("嘎嘣脆智能垃圾识别")
    child_ui = Ui_KidWindow()
    child_ui.setupUi(child)

    # 按钮绑定事件
    btn = main.pushButton_3
    btn.clicked.connect(child.show)

    #显示
    main.show()
    sys.exit(app.exec_())
