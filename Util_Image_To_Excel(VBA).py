Sub InsertPictures()

 Set fs = Application.FileSearch

 With fs
    .LookIn = ThisWorkbook.Path & "\"
    .Filename = "*.jpg"
    .Execute
 End With

 For i = 1 To fs.FoundFiles.Count

 With ActiveSheet.Pictures.Insert(fs.FoundFiles(i)).ShapeRange
    .LockAspectRatio = msoFalse
    .Height = Range("C1").Offset(i, 0).Height
    .Width = Range("C1").Offset(i, 0).Width
    .Left = Range("C1").Offset(i, 0).Left
    .Top = Range("C1").Offset(i, 0).Top
 End With
    Range("B1").Offset(i, 0) = OnlyFileName(fs.FoundFiles(i))
 Next i

End Sub

# ==============================

Function OnlyFileName(stFullName As String) As String

 Dim stSeparator As String
 Dim iLen As Integer
 Dim i As Integer

 stSeparator = Application.PathSeparator
 iLen = Len(stFullName)

 For i = iLen To 1 Step -1
    If Mid(stFullName, i, 1) = stSeparator Then Exit For
 Next i

 OnlyFileName = Left(Right(stFullName, iLen, - i), Len(Right(stFullName, iLen - i)) - 4)

End Function

# ==============================

Sub DeletePictures()
 With ActiveSheet
    .Pictures.Delete
    .Range(.Range("a2"), .Range("a2").End(xlDown)).ClearContents
 End With
End Sub
