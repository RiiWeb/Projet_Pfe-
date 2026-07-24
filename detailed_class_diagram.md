```mermaid
classDiagram
    class User {
        +Integer id
        +String username
        +String password_hash
        +String email
        +Boolean is_active
        +login()
        +logout()
    }
    User --|> UserMixin
    User --|> db.Model

    class Contact {
        +HTML content
        +CSS styling
        +render()
    }

    class Feedback {
        +HTML content
        +CSS styling
        +submit()
    }

    Contact --|> Templates
    Feedback --|> Templates

    class Templates {
        +HTML structure
        +CSS styling
        +render()
    }

    class FlaskApp {
        +String config
        +SQLAlchemy db
        +LoginManager login_manager
        +run()
    }

    FlaskApp --|> User
    FlaskApp --|> Templates

    class Database {
        +SQLite fraud.db
        +SQLite history.db
        +connect()
    }

    FlaskApp --|> Database