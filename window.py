# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'window.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.status_label = QtWidgets.QLabel(self.centralwidget)
        self.status_label.setObjectName("status_label")
        self.gridLayout.addWidget(self.status_label, 2, 0, 1, 1)
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.tab)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.groupBox = QtWidgets.QGroupBox(self.tab)
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.base_data_path = QtWidgets.QLineEdit(self.groupBox)
        self.base_data_path.setObjectName("base_data_path")
        self.horizontalLayout.addWidget(self.base_data_path)
        self.base_data_btn = QtWidgets.QPushButton(self.groupBox)
        self.base_data_btn.setObjectName("base_data_btn")
        self.horizontalLayout.addWidget(self.base_data_btn)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.item_data_path = QtWidgets.QLineEdit(self.groupBox)
        self.item_data_path.setObjectName("item_data_path")
        self.horizontalLayout_2.addWidget(self.item_data_path)
        self.item_data_btn = QtWidgets.QPushButton(self.groupBox)
        self.item_data_btn.setObjectName("item_data_btn")
        self.horizontalLayout_2.addWidget(self.item_data_btn)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.user_data_path = QtWidgets.QLineEdit(self.groupBox)
        self.user_data_path.setObjectName("user_data_path")
        self.horizontalLayout_4.addWidget(self.user_data_path)
        self.user_data_btn = QtWidgets.QPushButton(self.groupBox)
        self.user_data_btn.setObjectName("user_data_btn")
        self.horizontalLayout_4.addWidget(self.user_data_btn)
        self.verticalLayout_2.addLayout(self.horizontalLayout_4)
        self.load_data_btn = QtWidgets.QPushButton(self.groupBox)
        self.load_data_btn.setObjectName("load_data_btn")
        self.verticalLayout_2.addWidget(self.load_data_btn)
        self.verticalLayout_4.addWidget(self.groupBox)
        self.groupBox_2 = QtWidgets.QGroupBox(self.tab)
        self.groupBox_2.setObjectName("groupBox_2")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.groupBox_2)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self.label = QtWidgets.QLabel(self.groupBox_2)
        self.label.setObjectName("label")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label)
        self.filtering_methods_cb = QtWidgets.QComboBox(self.groupBox_2)
        self.filtering_methods_cb.setObjectName("filtering_methods_cb")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.filtering_methods_cb)
        self.verticalLayout_3.addLayout(self.formLayout)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.process_btn = QtWidgets.QPushButton(self.groupBox_2)
        self.process_btn.setMinimumSize(QtCore.QSize(0, 25))
        self.process_btn.setObjectName("process_btn")
        self.horizontalLayout_5.addWidget(self.process_btn)
        self.save_btn = QtWidgets.QPushButton(self.groupBox_2)
        self.save_btn.setMinimumSize(QtCore.QSize(0, 25))
        self.save_btn.setObjectName("save_btn")
        self.horizontalLayout_5.addWidget(self.save_btn)
        self.verticalLayout_3.addLayout(self.horizontalLayout_5)
        self.verticalLayout_4.addWidget(self.groupBox_2)
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.tab_2)
        self.verticalLayout.setObjectName("verticalLayout")
        self.groupBox_3 = QtWidgets.QGroupBox(self.tab_2)
        self.groupBox_3.setObjectName("groupBox_3")
        self.formLayout_2 = QtWidgets.QFormLayout(self.groupBox_3)
        self.formLayout_2.setObjectName("formLayout_2")
        self.label_2 = QtWidgets.QLabel(self.groupBox_3)
        self.label_2.setObjectName("label_2")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.matrix_file_path = QtWidgets.QLineEdit(self.groupBox_3)
        self.matrix_file_path.setObjectName("matrix_file_path")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.matrix_file_path)
        self.load_matrix_btn = QtWidgets.QPushButton(self.groupBox_3)
        self.load_matrix_btn.setObjectName("load_matrix_btn")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.SpanningRole, self.load_matrix_btn)
        self.verticalLayout.addWidget(self.groupBox_3)
        self.groupBox_4 = QtWidgets.QGroupBox(self.tab_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_4.sizePolicy().hasHeightForWidth())
        self.groupBox_4.setSizePolicy(sizePolicy)
        self.groupBox_4.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.groupBox_4.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.groupBox_4.setObjectName("groupBox_4")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox_4)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.tabWidget_2 = QtWidgets.QTabWidget(self.groupBox_4)
        self.tabWidget_2.setObjectName("tabWidget_2")
        self.DBScan = QtWidgets.QWidget()
        self.DBScan.setObjectName("DBScan")
        self.groupBox_5 = QtWidgets.QGroupBox(self.DBScan)
        self.groupBox_5.setGeometry(QtCore.QRect(10, 10, 206, 133))
        self.groupBox_5.setObjectName("groupBox_5")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.groupBox_5)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.formLayout_3 = QtWidgets.QFormLayout()
        self.formLayout_3.setObjectName("formLayout_3")
        self.label_3 = QtWidgets.QLabel(self.groupBox_5)
        self.label_3.setObjectName("label_3")
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_3)
        self.eps_edit = QtWidgets.QLineEdit(self.groupBox_5)
        self.eps_edit.setObjectName("eps_edit")
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.eps_edit)
        self.label_4 = QtWidgets.QLabel(self.groupBox_5)
        self.label_4.setObjectName("label_4")
        self.formLayout_3.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_4)
        self.min_points_edit = QtWidgets.QLineEdit(self.groupBox_5)
        self.min_points_edit.setObjectName("min_points_edit")
        self.formLayout_3.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.min_points_edit)
        self.gridLayout_3.addLayout(self.formLayout_3, 0, 0, 1, 1)
        self.run_dbscan_btn = QtWidgets.QPushButton(self.groupBox_5)
        self.run_dbscan_btn.setObjectName("run_dbscan_btn")
        self.gridLayout_3.addWidget(self.run_dbscan_btn, 1, 0, 1, 1)
        self.tabWidget_2.addTab(self.DBScan, "")
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")
        self.tabWidget_2.addTab(self.tab_4, "")
        self.gridLayout_2.addWidget(self.tabWidget_2, 0, 0, 1, 1)
        self.verticalLayout.addWidget(self.groupBox_4)
        self.tabWidget.addTab(self.tab_2, "")
        self.gridLayout.addWidget(self.tabWidget, 1, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.status_label.setText(_translate("MainWindow", ":D"))
        self.groupBox.setTitle(_translate("MainWindow", "Load DaTaSet"))
        self.base_data_btn.setText(_translate("MainWindow", "base data"))
        self.item_data_btn.setText(_translate("MainWindow", "item data"))
        self.user_data_btn.setText(_translate("MainWindow", "user data"))
        self.load_data_btn.setText(_translate("MainWindow", "load data"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Apply Filtering Technigue"))
        self.label.setText(_translate("MainWindow", "TextLabel"))
        self.process_btn.setText(_translate("MainWindow", "Process"))
        self.save_btn.setText(_translate("MainWindow", "Save"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Tab 1"))
        self.groupBox_3.setTitle(_translate("MainWindow", "Load distance matrix"))
        self.label_2.setText(_translate("MainWindow", "Load distance matrix :"))
        self.load_matrix_btn.setText(_translate("MainWindow", "Search"))
        self.groupBox_4.setTitle(_translate("MainWindow", "Clustering Algorithms"))
        self.groupBox_5.setTitle(_translate("MainWindow", "Parameters"))
        self.label_3.setText(_translate("MainWindow", "EPS :"))
        self.label_4.setText(_translate("MainWindow", "Min Points :"))
        self.run_dbscan_btn.setText(_translate("MainWindow", "Run"))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.DBScan), _translate("MainWindow", "Tab 1"))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab_4), _translate("MainWindow", "Tab 2"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Tab 2"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

