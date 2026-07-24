```mermaid
classDiagram
    class User {
        +Integer id
    }
    User --|> UserMixin
    User --|> db.Model

    class Contact {
        +HTML content
    }

    class Feedback {
        +HTML content
    }

    Contact --|> Templates
    Feedback --|> Templates