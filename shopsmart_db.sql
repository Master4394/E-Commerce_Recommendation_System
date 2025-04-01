-- Step 1: Create the database
CREATE DATABASE IF NOT EXISTS shopsmart_db;
USE shopsmart_db;

-- Step 2: Create tables
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) NOT NULL DEFAULT 'customer',
    phone VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS products (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    brand VARCHAR(100) NOT NULL,
    category VARCHAR(100) NOT NULL,
    description TEXT NOT NULL,
    price DECIMAL(10, 2) NOT NULL,
    rating DECIMAL(3, 1) DEFAULT 0,
    reviews_count INT DEFAULT 0,
    stock INT NOT NULL,
    image_url VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS orders (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    total DECIMAL(10, 2) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS order_items (
    id INT AUTO_INCREMENT PRIMARY KEY,
    order_id INT NOT NULL,
    product_id INT NOT NULL,
    quantity INT NOT NULL,
    price DECIMAL(10, 2) NOT NULL,
    FOREIGN KEY (order_id) REFERENCES orders(id),
    FOREIGN KEY (product_id) REFERENCES products(id)
);

CREATE TABLE IF NOT EXISTS wishlist (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    product_id INT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (product_id) REFERENCES products(id),
    UNIQUE(user_id, product_id)
);

CREATE TABLE IF NOT EXISTS user_interactions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    product_id INT NOT NULL,
    interaction_type VARCHAR(20) NOT NULL, 
    weight DECIMAL(3, 2) NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (product_id) REFERENCES products(id)
);

CREATE TABLE IF NOT EXISTS categories (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS user_preferences (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    preference_key VARCHAR(100) NOT NULL,
    preference_value TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id),
    UNIQUE(user_id, preference_key)
);

-- Step 3: Insert Default Data
INSERT INTO categories (name) 
SELECT * FROM (SELECT 'electronics') AS tmp 
WHERE NOT EXISTS (SELECT name FROM categories WHERE name = 'electronics') LIMIT 1;

INSERT INTO categories (name) 
SELECT * FROM (SELECT 'clothing') AS tmp 
WHERE NOT EXISTS (SELECT name FROM categories WHERE name = 'clothing') LIMIT 1;

INSERT INTO categories (name) 
SELECT * FROM (SELECT 'home') AS tmp 
WHERE NOT EXISTS (SELECT name FROM categories WHERE name = 'home') LIMIT 1;

INSERT INTO categories (name) 
SELECT * FROM (SELECT 'books') AS tmp 
WHERE NOT EXISTS (SELECT name FROM categories WHERE name = 'books') LIMIT 1;

INSERT INTO categories (name) 
SELECT * FROM (SELECT 'beauty') AS tmp 
WHERE NOT EXISTS (SELECT name FROM categories WHERE name = 'beauty') LIMIT 1;

INSERT INTO categories (name) 
SELECT * FROM (SELECT 'sports') AS tmp 
WHERE NOT EXISTS (SELECT name FROM categories WHERE name = 'sports') LIMIT 1;

INSERT INTO users (first_name, last_name, email, password_hash, role)
SELECT * FROM (SELECT 'Admin', 'User', 'admin@example.com', SHA2('admin123', 256), 'admin') AS tmp
WHERE NOT EXISTS (SELECT email FROM users WHERE email = 'admin@example.com') LIMIT 1;
