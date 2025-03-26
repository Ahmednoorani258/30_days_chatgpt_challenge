class Book:
    def __init__(self, title, author):
        self.title = title
        self.author = author
        self.is_borrowed = False

class Library:
    def __init__(self):
        self.books = []
    # Add methods: add_book, borrow_book, return_book
    
    def add_books(self, book):
        self.books.append(book)
    
    def borrow_book(self, title):
        for book in self.books:
            if book.title == title and not book.is_borrowed:
                book.is_borrowed = True
                print(f"Book '{title}' borrowed.")
                return
        print(f"Book '{title}' not found or already borrowed.")
    
    def return_book(self, title):
        for book in self.books:
            if book.title == title and book.is_borrowed:
                book.is_borrowed = False
                print(f"Book '{title}' returned.")
                return
        print(f"Book '{title}' not found or not borrowed.")
    
    def display_books(self):
        if not self.books:
            print("No books in the library.")
        else:
            for book in self.books:
                print(f"Title: {book.title}, Author: {book.author}, Borrowed: {book.is_borrowed}")

# Usage
library = Library()

book1 = Book("The Great Gatsby", "F. Scott Fitzgerald")
book2 = Book("To Kill a Mockingbird", "Harper Lee")

print("\n")
print("\n")
library.add_books(book1)
library.add_books(book2)
print("\n")
library.display_books()
print("\n")

library.borrow_book("The Great Gatsby")
print("\n")

library.display_books()
print("\n")

library.return_book("To Kill a Mockingbird")