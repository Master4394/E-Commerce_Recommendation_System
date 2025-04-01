from flask import Flask, request, jsonify, render_template, session, redirect, url_for, flash
from flask_mysqldb import MySQL
from werkzeug.security import generate_password_hash, check_password_hash
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import secrets
import re
from functools import wraps

app = Flask(__name__)
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)


# Configuration
app.secret_key = secrets.token_hex(16)
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'your_password'
app.config['MYSQL_DB'] = 'shopsmart_db'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

# Initialize MySQL
mysql = MySQL(app)

# Recommendation models
content_based_model = None
collaborative_model = None

# Authentication decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({"error": "Authentication required"}), 401
        return f(*args, **kwargs)
    return decorated_function

# Admin decorator
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({"error": "Authentication required"}), 401
        
        # Check if user is admin
        cur = mysql.connection.cursor()
        cur.execute("SELECT role FROM users WHERE id = %s", [session['user_id']])
        user = cur.fetchone()
        cur.close()
        
        if user['role'] != 'admin':
            return jsonify({"error": "Admin privileges required"}), 403
            
        return f(*args, **kwargs)
    return decorated_function

# Database initialization
@app.before_request
def initialize_database():
    try:
        # Establish a new database connection
        conn = mysql.connection
        cur = conn.cursor()
        
    except Exception as e:
        print("Error initializing database:", e)
    # Create users table
    cur.execute('''
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
    )
    ''')
    
    # Create products table
    cur.execute('''
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
    )
    ''')
    
    # Create orders table
    cur.execute('''
    CREATE TABLE IF NOT EXISTS orders (
        id INT AUTO_INCREMENT PRIMARY KEY,
        user_id INT NOT NULL,
        total DECIMAL(10, 2) NOT NULL,
        status VARCHAR(50) NOT NULL DEFAULT 'pending',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
    ''')
    
    # Create order_items table
    cur.execute('''
    CREATE TABLE IF NOT EXISTS order_items (
        id INT AUTO_INCREMENT PRIMARY KEY,
        order_id INT NOT NULL,
        product_id INT NOT NULL,
        quantity INT NOT NULL,
        price DECIMAL(10, 2) NOT NULL,
        FOREIGN KEY (order_id) REFERENCES orders(id),
        FOREIGN KEY (product_id) REFERENCES products(id)
    )
    ''')
    
    # Create wishlist table
    cur.execute('''
    CREATE TABLE IF NOT EXISTS wishlist (
        id INT AUTO_INCREMENT PRIMARY KEY,
        user_id INT NOT NULL,
        product_id INT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id),
        FOREIGN KEY (product_id) REFERENCES products(id),
        UNIQUE(user_id, product_id)
    )
    ''')
    
    # Create user_interactions table (for recommendation system)
    cur.execute('''
    CREATE TABLE IF NOT EXISTS user_interactions (
        id INT AUTO_INCREMENT PRIMARY KEY,
        user_id INT NOT NULL,
        product_id INT NOT NULL,
        interaction_type VARCHAR(20) NOT NULL, 
        weight DECIMAL(3, 2) NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id),
        FOREIGN KEY (product_id) REFERENCES products(id)
    )
    ''')
    
    # Create product categories for filtering
    cur.execute('''
    CREATE TABLE IF NOT EXISTS categories (
        id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(100) UNIQUE NOT NULL
    )
    ''')
    
    # Create user preferences table for personalized recommendations
    cur.execute('''
    CREATE TABLE IF NOT EXISTS user_preferences (
        id INT AUTO_INCREMENT PRIMARY KEY,
        user_id INT NOT NULL,
        preference_key VARCHAR(100) NOT NULL,
        preference_value TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users(id),
        UNIQUE(user_id, preference_key)
    )
    ''')
    
    # Insert default categories if not exist
    categories = ['electronics', 'clothing', 'home', 'books', 'beauty', 'sports']
    for category in categories:
        cur.execute("SELECT * FROM categories WHERE name = %s", [category])
        if not cur.fetchone():
            cur.execute("INSERT INTO categories (name) VALUES (%s)", [category])
    
    # Insert demo admin user if not exists
    cur.execute("SELECT * FROM users WHERE email = %s", ['admin@example.com'])
    if not cur.fetchone():
        password_hash = generate_password_hash('admin123')
        cur.execute('''
        INSERT INTO users (first_name, last_name, email, password_hash, role)
        VALUES (%s, %s, %s, %s, %s)
        ''', ['Admin', 'User', 'admin@example.com', password_hash, 'admin'])
    
    # Insert demo products if none exist
    cur.execute("SELECT COUNT(*) as count FROM products")
    product_count = cur.fetchone()['count']
    
    if product_count == 0:
        # Add demo products (sample data)
        demo_products = [
            {
                'name': 'Smart 4K TV',
                'brand': 'TechPro',
                'category': 'electronics',
                'price': 499.99,
                'rating': 4.7,
                'reviews_count': 245,
                'stock': 12,
                'description': 'Ultra HD Smart TV with built-in streaming services and voice control.',
                'image_url': 'https://picsum.photos/seed/tv1/300/200'
            },
            {
                'name': 'Wireless Noise-Canceling Headphones',
                'brand': 'AudioMax',
                'category': 'electronics',
                'price': 199.99,
                'rating': 4.8,
                'reviews_count': 187,
                'stock': 25,
                'description': 'Premium wireless headphones with active noise cancellation and 30-hour battery life.',
                'image_url': url_for('static', filename='images/1.jpg')
            },
            # Add more sample products as needed
        ]
        
        for product in demo_products:
            cur.execute('''
            INSERT INTO products (name, brand, category, price, rating, reviews_count, stock, description, image_url)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ''', [
                product['name'], 
                product['brand'], 
                product['category'], 
                product['price'], 
                product['rating'], 
                product['reviews_count'], 
                product['stock'], 
                product['description'], 
                product['image_url']
            ])
    
    mysql.connection.commit()
    cur.close()
    
    # Initialize recommendation models
    initialize_recommendation_models()

# Initialize recommendation models
def initialize_recommendation_models():
    global content_based_model, collaborative_model
    
    try:
        # Load products data
        cur = mysql.connection.cursor()
        cur.execute('''
        SELECT p.*, COUNT(ui.id) as interaction_count
        FROM products p
        LEFT JOIN user_interactions ui ON p.id = ui.product_id
        GROUP BY p.id
        ''')
        products = cur.fetchall()
        
        # Load user interactions for collaborative filtering
        cur.execute('''
        SELECT user_id, product_id, interaction_type, weight
        FROM user_interactions
        ''')
        interactions = cur.fetchall()
        cur.close()
        
        if products:
            # Create content-based model
            # Combine relevant features for content-based filtering
            products_df = pd.DataFrame(products)
            if len(products_df) > 0:
                products_df['features'] = products_df.apply(
                    lambda x: f"{x['name']} {x['brand']} {x['category']} {x['description']}", 
                    axis=1
                )
                
                # Create TF-IDF vectorizer
                tfidf = TfidfVectorizer(
                    stop_words='english',
                    max_features=5000
                )
                
                tfidf_matrix = tfidf.fit_transform(products_df['features'])
                content_based_model = {
                    'tfidf': tfidf,
                    'matrix': tfidf_matrix,
                    'products': products_df
                }
                
                # Create collaborative model if we have enough interactions
                if interactions and len(interactions) > 10:
                    interactions_df = pd.DataFrame(interactions)
                    
                    # Create user-item matrix
                    user_item_df = interactions_df.pivot_table(
                        index='user_id',
                        columns='product_id',
                        values='weight',
                        aggfunc='mean',
                        fill_value=0
                    )
                    
                    # Apply matrix factorization (SVD)
                    svd = TruncatedSVD(n_components=min(10, user_item_df.shape[1]-1))
                    latent_features = svd.fit_transform(user_item_df)
                    
                    collaborative_model = {
                        'svd': svd,
                        'features': latent_features,
                        'user_map': {user: i for i, user in enumerate(user_item_df.index)},
                        'user_item_matrix': user_item_df
                    }
                    
        app.logger.info("Recommendation models initialized successfully")
    except Exception as e:
        app.logger.error(f"Error initializing recommendation models: {e}")

# Routes for frontend
@app.route('/')
def index():
    return render_template('index.html')

# Authentication routes
@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    
    # Validate required fields
    required_fields = ['first_name', 'last_name', 'email', 'password']
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400
    
    # Validate email format
    email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_regex, data['email']):
        return jsonify({"error": "Invalid email format"}), 400
    
    # Check if email already exists
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM users WHERE email = %s", [data['email']])
    if cur.fetchone():
        cur.close()
        return jsonify({"error": "Email already in use"}), 400
    
    # Create new user
    password_hash = generate_password_hash(data['password'])
    role = data.get('role', 'customer')  # Only admins can set role
    
    # Validate role
    if role not in ['customer', 'admin']:
        role = 'customer'  # Default to customer if invalid role
    
    try:
        cur.execute('''
        INSERT INTO users (first_name, last_name, email, password_hash, role, phone)
        VALUES (%s, %s, %s, %s, %s, %s)
        ''', [
            data['first_name'],
            data['last_name'],
            data['email'],
            password_hash,
            role,
            data.get('phone', None)
        ])
        mysql.connection.commit()
        
        # Get the new user ID
        user_id = cur.lastrowid
        cur.close()
        
        # Create session
        session['user_id'] = user_id
        session['role'] = role
        
        return jsonify({
            "success": True,
            "user": {
                "id": user_id,
                "first_name": data['first_name'],
                "last_name": data['last_name'],
                "email": data['email'],
                "role": role
            }
        }), 201
    except Exception as e:
        cur.close()
        return jsonify({"error": str(e)}), 500

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    
    # Validate required fields
    if 'email' not in data or 'password' not in data:
        return jsonify({"error": "Email and password are required"}), 400
    
    # Check user credentials
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM users WHERE email = %s", [data['email']])
    user = cur.fetchone()
    cur.close()
    
    if not user or not check_password_hash(user['password_hash'], data['password']):
        return jsonify({"error": "Invalid email or password"}), 401
    
    # Update last_active timestamp
    cur = mysql.connection.cursor()
    cur.execute("UPDATE users SET last_active = NOW() WHERE id = %s", [user['id']])
    mysql.connection.commit()
    cur.close()
    
    # Create session
    session['user_id'] = user['id']
    session['role'] = user['role']
    
    return jsonify({
        "success": True,
        "user": {
            "id": user['id'],
            "first_name": user['first_name'],
            "last_name": user['last_name'],
            "email": user['email'],
            "role": user['role'],
            "created_at": user['created_at'].isoformat() if user['created_at'] else None
        }
    }), 200

@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({"success": True}), 200

@app.route('/api/user', methods=['GET'])
@login_required
def get_current_user():
    cur = mysql.connection.cursor()
    cur.execute("SELECT id, first_name, last_name, email, role, phone, created_at, last_active FROM users WHERE id = %s", [session['user_id']])
    user = cur.fetchone()
    cur.close()
    
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    return jsonify({
        "id": user['id'],
        "first_name": user['first_name'],
        "last_name": user['last_name'],
        "email": user['email'],
        "role": user['role'],
        "phone": user['phone'],
        "created_at": user['created_at'].isoformat() if user['created_at'] else None,
        "last_active": user['last_active'].isoformat() if user['last_active'] else None
    }), 200

@app.route('/api/user', methods=['PUT'])
@login_required
def update_user():
    data = request.get_json()
    allowed_fields = ['first_name', 'last_name', 'email', 'phone']
    
    # Build update query dynamically with only allowed fields
    update_fields = []
    params = []
    
    for field in allowed_fields:
        if field in data:
            update_fields.append(f"{field} = %s")
            params.append(data[field])
    
    if not update_fields:
        return jsonify({"error": "No valid fields to update"}), 400
    
    # Add user_id to params
    params.append(session['user_id'])
    
    try:
        cur = mysql.connection.cursor()
        query = f"UPDATE users SET {', '.join(update_fields)} WHERE id = %s"
        cur.execute(query, params)
        mysql.connection.commit()
        cur.close()
        
        return jsonify({"success": True}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/user/password', methods=['PUT'])
@login_required
def update_password():
    data = request.get_json()
    
    if 'current_password' not in data or 'new_password' not in data:
        return jsonify({"error": "Current password and new password are required"}), 400
    
    # Verify current password
    cur = mysql.connection.cursor()
    cur.execute("SELECT password_hash FROM users WHERE id = %s", [session['user_id']])
    user = cur.fetchone()
    
    if not check_password_hash(user['password_hash'], data['current_password']):
        cur.close()
        return jsonify({"error": "Current password is incorrect"}), 401
    
    # Update password
    new_password_hash = generate_password_hash(data['new_password'])
    cur.execute("UPDATE users SET password_hash = %s WHERE id = %s", [new_password_hash, session['user_id']])
    mysql.connection.commit()
    cur.close()
    
    return jsonify({"success": True}), 200

# Product routes
@app.route('/api/products', methods=['GET'])
def get_products():
    # Parse query parameters
    category = request.args.get('category')
    min_price = request.args.get('min_price')
    max_price = request.args.get('max_price')
    sort = request.args.get('sort', 'popularity')
    page = int(request.args.get('page', 1))
    limit = int(request.args.get('limit', 20))
    search = request.args.get('search')
    
    # Build query
    query = "SELECT * FROM products WHERE 1=1"
    params = []
    
    if category and category != 'all':
        query += " AND category = %s"
        params.append(category)
    
    if min_price:
        query += " AND price >= %s"
        params.append(float(min_price))
    
    if max_price:
        query += " AND price <= %s"
        params.append(float(max_price))
    
    if search:
        query += " AND (name LIKE %s OR brand LIKE %s OR description LIKE %s)"
        search_term = f"%{search}%"
        params.extend([search_term, search_term, search_term])
    
    # Add sorting
    if sort == 'price_low':
        query += " ORDER BY price ASC"
    elif sort == 'price_high':
        query += " ORDER BY price DESC"
    elif sort == 'rating':
        query += " ORDER BY rating DESC"
    elif sort == 'newest':
        query += " ORDER BY created_at DESC"
    else:  # default to popularity (reviews_count)
        query += " ORDER BY reviews_count DESC"
    
    # Add pagination
    offset = (page - 1) * limit
    query += " LIMIT %s OFFSET %s"
    params.extend([limit, offset])
    
    # Execute query
    cur = mysql.connection.cursor()
    cur.execute(query, params)
    products = cur.fetchall()
    
    # Get total count for pagination
    count_query = "SELECT COUNT(*) as total FROM products WHERE 1=1"
    count_params = []
    
    if category and category != 'all':
        count_query += " AND category = %s"
        count_params.append(category)
    
    if min_price:
        count_query += " AND price >= %s"
        count_params.append(float(min_price))
    
    if max_price:
        count_query += " AND price <= %s"
        count_params.append(float(max_price))
    
    if search:
        count_query += " AND (name LIKE %s OR brand LIKE %s OR description LIKE %s)"
        search_term = f"%{search}%"
        count_params.extend([search_term, search_term, search_term])
    
    cur.execute(count_query, count_params)
    total = cur.fetchone()['total']
    cur.close()
    
    # Record user interactions if user is logged in and search/filtering is happening
    if 'user_id' in session and (search or category != 'all'):
        record_search_interaction(session['user_id'], search, category)
    
    # Calculate pagination info
    total_pages = (total + limit - 1) // limit
    
    return jsonify({
        "products": products,
        "pagination": {
            "page": page,
            "limit": limit,
            "total": total,
            "total_pages": total_pages
        }
    }), 200

@app.route('/api/products/<int:product_id>', methods=['GET'])
def get_product(product_id):
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM products WHERE id = %s", [product_id])
    product = cur.fetchone()
    cur.close()
    
    if not product:
        return jsonify({"error": "Product not found"}), 404
    
    # Record view interaction if user is logged in
    if 'user_id' in session:
        record_product_interaction(session['user_id'], product_id, 'view', 1.0)
    
    return jsonify(product), 200

@app.route('/api/products', methods=['POST'])
@admin_required
def create_product():
    data = request.get_json()
    
    # Validate required fields
    required_fields = ['name', 'brand', 'category', 'description', 'price', 'stock', 'image_url']
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400
    
    # Validate numeric fields
    try:
        price = float(data['price'])
        stock = int(data['stock'])
        rating = float(data.get('rating', 0))
        reviews_count = int(data.get('reviews_count', 0))
    except ValueError:
        return jsonify({"error": "Invalid numeric field"}), 400
    
    # Insert product
    try:
        cur = mysql.connection.cursor()
        cur.execute('''
        INSERT INTO products (name, brand, category, description, price, stock, rating, reviews_count, image_url)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ''', [
            data['name'],
            data['brand'],
            data['category'],
            data['description'],
            price,
            stock,
            rating,
            reviews_count,
            data['image_url']
        ])
        mysql.connection.commit()
        
        # Get the new product ID
        product_id = cur.lastrowid
        cur.close()
        
        # Update recommendation models
        initialize_recommendation_models()
        
        return jsonify({
            "success": True,
            "product_id": product_id
        }), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/products/<int:product_id>', methods=['PUT'])
@admin_required
def update_product(product_id):
    data = request.get_json()
    
    # Check if product exists
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM products WHERE id = %s", [product_id])
    if not cur.fetchone():
        cur.close()
        return jsonify({"error": "Product not found"}), 404
    
    # Build update query dynamically
    allowed_fields = ['name', 'brand', 'category', 'description', 'price', 'stock', 'rating', 'reviews_count', 'image_url']
    update_fields = []
    params = []
    
    for field in allowed_fields:
        if field in data:
            update_fields.append(f"{field} = %s")
            params.append(data[field])
    
    if not update_fields:
        cur.close()
        return jsonify({"error": "No valid fields to update"}), 400
    
    # Add product_id to params
    params.append(product_id)
    
    try:
        query = f"UPDATE products SET {', '.join(update_fields)} WHERE id = %s"
        cur.execute(query, params)
        mysql.connection.commit()
        cur.close()
        
        # Update recommendation models
        initialize_recommendation_models()
        
        return jsonify({"success": True}), 200
    except Exception as e:
        cur.close()
        return jsonify({"error": str(e)}), 500

@app.route('/api/products/<int:product_id>', methods=['DELETE'])
@admin_required
def delete_product(product_id):
    # Check if product exists
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM products WHERE id = %s", [product_id])
    if not cur.fetchone():
        cur.close()
        return jsonify({"error": "Product not found"}), 404
    
    try:
        # Delete product
        cur.execute("DELETE FROM products WHERE id = %s", [product_id])
        mysql.connection.commit()
        cur.close()
        
        # Update recommendation models
        initialize_recommendation_models()
        
        return jsonify({"success": True}), 200
    except Exception as e:
        cur.close()
        return jsonify({"error": str(e)}), 500

# Category routes
@app.route('/api/categories', methods=['GET'])
def get_categories():
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM categories")
    categories = cur.fetchall()
    cur.close()
    
    return jsonify(categories), 200

# Wishlist routes
@app.route('/api/wishlist', methods=['GET'])
@login_required
def get_wishlist():
    cur = mysql.connection.cursor()
    cur.execute('''
    SELECT p.*, w.created_at as added_at
    FROM wishlist w
    JOIN products p ON w.product_id = p.id
    WHERE w.user_id = %s
    ORDER BY w.created_at DESC
    ''', [session['user_id']])
    wishlist = cur.fetchall()
    cur.close()
    
    return jsonify(wishlist), 200

@app.route('/api/wishlist/<int:product_id>', methods=['POST'])
@login_required
def add_to_wishlist(product_id):
    # Check if product exists
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM products WHERE id = %s", [product_id])
    if not cur.fetchone():
        cur.close()
        return jsonify({"error": "Product not found"}), 404
    
    # Check if already in wishlist
    cur.execute("SELECT * FROM wishlist WHERE user_id = %s AND product_id = %s", 
                [session['user_id'], product_id])
    if cur.fetchone():
        cur.close()
        return jsonify({"error": "Product already in wishlist"}), 400
    
    try:
        # Add to wishlist
        cur.execute('''
        INSERT INTO wishlist (user_id, product_id)
        VALUES (%s, %s)
        ''', [session['user_id'], product_id])
        mysql.connection.commit()
        
        # Record interaction
        record_product_interaction(session['user_id'], product_id, 'wishlist', 0.8)
        
        cur.close()
        return jsonify({"success": True}), 201
    except Exception as e:
        cur.close()
        return jsonify({"error": str(e)}), 500

@app.route('/api/wishlist/<int:product_id>', methods=['DELETE'])
@login_required
def remove_from_wishlist(product_id):
    try:
        cur = mysql.connection.cursor()
        cur.execute('''
        DELETE FROM wishlist 
        WHERE user_id = %s AND product_id = %s
        ''', [session['user_id'], product_id])
        mysql.connection.commit()
        cur.close()
        
        return jsonify({"success": True}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Orders routes
@app.route('/api/orders', methods=['GET'])
@login_required
def get_orders():
    cur = mysql.connection.cursor()
    cur.execute('''
    SELECT o.*, COUNT(oi.id) as item_count
    FROM orders o
    LEFT JOIN order_items oi ON o.id = oi.order_id
    WHERE o.user_id = %s
    GROUP BY o.id
    ORDER BY o.created_at DESC
    ''', [session['user_id']])
    orders = cur.fetchall()
    cur.close()
    
    return jsonify(orders), 200

@app.route('/api/orders/<int:order_id>', methods=['GET'])
@login_required
def get_order(order_id):
    cur = mysql.connection.cursor()
    
    # Get order details
    cur.execute('''
    SELECT * FROM orders 
    WHERE id = %s AND user_id = %s
    ''', [order_id, session['user_id']])
    order = cur.fetchone()
    
    if not order:
        cur.close()
        return jsonify({"error": "Order not found"}), 404
    
    # Get order items
    cur.execute('''
    SELECT oi.*, p.name, p.brand, p.image_url
    FROM order_items oi
    JOIN products p ON oi.product_id = p.id
    WHERE oi.order_id = %s
    ''', [order_id])
    items = cur.fetchall()
    
    cur.close()
    
    return jsonify({
        "order": order,
        "items": items
    }), 200

@app.route('/api/orders', methods=['POST'])
@login_required
def create_order():
    data = request.get_json()
    
    if 'items' not in data or not data['items']:
        return jsonify({"error": "Order must contain items"}), 400
    
    # Validate items format
    for item in data['items']:
        if 'product_id' not in item or 'quantity' not in item:
            return jsonify({"error": "Each item must have product_id and quantity"}), 400
        
        # Check if product exists and has enough stock
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM products WHERE id = %s", [item['product_id']])
        product = cur.fetchone()
        cur.close()
        
        if not product:
            return jsonify({"error": f"Product {item['product_id']} not found"}), 404
        
        if product['stock'] < item['quantity']:
            return jsonify({"error": f"Not enough stock for product {product['name']}"}), 400
    
    # Create order
    try:
        # Start transaction
        cur = mysql.connection.cursor()
        cur.execute("START TRANSACTION")
        
        # Calculate total
        total = 0
        for item in data['items']:
            cur.execute("SELECT price FROM products WHERE id = %s", [item['product_id']])
            product = cur.fetchone()
            total += product['price'] * item['quantity']
        
        # Create order
        cur.execute('''
        INSERT INTO orders (user_id, total, status)
        VALUES (%s, %s, %s)
        ''', [session['user_id'], total, 'pending'])
        
        order_id = cur.lastrowid
        
        # Add order items
        for item in data['items']:
            cur.execute("SELECT price FROM products WHERE id = %s", [item['product_id']])
            product = cur.fetchone()
            
            cur.execute('''
            INSERT INTO order_items (order_id, product_id, quantity, price)
            VALUES (%s, %s, %s, %s)
            ''', [order_id, item['product_id'], item['quantity'], product['price']])
            
            # Update product stock
            cur.execute('''
            UPDATE products
            SET stock = stock - %s
            WHERE id = %s
            ''', [item['quantity'], item['product_id']])
            
            # Record purchase interaction
            record_product_interaction(session['user_id'], item['product_id'], 'purchase', 1.0)
        
        # Commit transaction
        mysql.connection.commit()
        cur.close()
        
        return jsonify({
            "success": True,
            "order_id": order_id,
            "total": total
        }), 201
    except Exception as e:
        # Rollback transaction on error
        cur.execute("ROLLBACK")
        cur.close()
        return jsonify({"error": str(e)}), 500

# Admin routes
@app.route('/api/admin/dashboard', methods=['GET'])
@admin_required
def admin_dashboard():
    cur = mysql.connection.cursor()
    
    # Get total sales
    cur.execute("SELECT SUM(total) as total_sales FROM orders")
    total_sales = cur.fetchone()['total_sales'] or 0
    
    # Get total orders
    cur.execute("SELECT COUNT(*) as total_orders FROM orders")
    total_orders = cur.fetchone()['total_orders']
    
    # Get active users
    cur.execute("SELECT COUNT(*) as active_users FROM users WHERE last_active > DATE_SUB(NOW(), INTERVAL 30 DAY)")
    active_users = cur.fetchone()['active_users']
    
    # Get conversion rate
    cur.execute('''
    SELECT 
        COUNT(DISTINCT user_id) as converted_users,
        (SELECT COUNT(*) FROM users) as total_users
    FROM orders
    ''')
    result = cur.fetchone()
    conversion_rate = (result['converted_users'] / result['total_users']) * 100 if result['total_users'] > 0 else 0
    
    # Get recent orders
    cur.execute('''
    SELECT o.*, u.first_name, u.last_name, COUNT(oi.id) as item_count
    FROM orders o
    JOIN users u ON o.user_id = u.id
    LEFT JOIN order_items oi ON o.id = oi.order_id
    GROUP BY o.id
    ORDER BY o.created_at DESC
    LIMIT 5
    ''')
    recent_orders = cur.fetchall()
    
    # Get monthly sales data for chart
    cur.execute('''
    SELECT 
        DATE_FORMAT(created_at, '%Y-%m') as month,
        SUM(total) as sales
    FROM orders
    WHERE created_at >= DATE_SUB(NOW(), INTERVAL 12 MONTH)
    GROUP BY month
    ORDER BY month
    ''')
    monthly_sales = cur.fetchall()
    
    # Get recommendation engine performance
    cur.execute('''
    SELECT 
        interaction_type,
        COUNT(*) as count
    FROM user_interactions
    WHERE timestamp >= DATE_SUB(NOW(), INTERVAL 30 DAY)
    GROUP BY interaction_type
    ''')
    interactions = {row['interaction_type']: row['count'] for row in cur.fetchall()}
    
    # Calculate CTR and conversion metrics
    total_views = interactions.get('view', 0)
    total_clicks = interactions.get('detail_view', 0) + interactions.get('cart', 0)
    total_purchases = interactions.get('purchase', 0)
    
    ctr = (total_clicks / total_views) * 100 if total_views > 0 else 0
    purchase_conversion = (total_purchases / total_clicks) * 100 if total_clicks > 0 else 0
    
    cur.close()
    
    return jsonify({
        "total_sales": total_sales,
        "total_orders": total_orders,
        "active_users": active_users,
        "conversion_rate": conversion_rate,
        "recent_orders": recent_orders,
        "monthly_sales": monthly_sales,
        "recommendation_metrics": {
            "ctr": ctr,
            "conversion_rate": purchase_conversion,
            "interactions": interactions
        }
    }), 200

@app.route('/api/admin/users', methods=['GET'])
@admin_required
def admin_get_users():
    cur = mysql.connection.cursor()
    cur.execute('''
    SELECT id, first_name, last_name, email, role, created_at, last_active
    FROM users
    ORDER BY id DESC
    ''')
    users = cur.fetchall()
    cur.close()
    
    return jsonify(users), 200

# Recommendation System Routes
@app.route('/api/recommendations/personalized', methods=['GET'])
@login_required
def get_personalized_recommendations():
    user_id = session['user_id']
    limit = int(request.args.get('limit', 8))
    
    # Get recommendations using both methods
    content_recommendations = get_content_based_recommendations(user_id, limit)
    collaborative_recommendations = get_collaborative_recommendations(user_id, limit)
    
    # Combine and deduplicate recommendations
    all_recs = content_recommendations + collaborative_recommendations
    unique_recs = []
    seen_ids = set()
    
    for rec in all_recs:
        if rec['id'] not in seen_ids:
            unique_recs.append(rec)
            seen_ids.add(rec['id'])
            if len(unique_recs) >= limit:
                break
    
    return jsonify(unique_recs), 200

@app.route('/api/recommendations/similar/<int:product_id>', methods=['GET'])
def get_similar_products(product_id):
    limit = int(request.args.get('limit', 8))
    
    # Get similar products using content-based filtering
    similar_products = get_content_based_similar_products(product_id, limit)
    
    # Record interaction if user is logged in
    if 'user_id' in session:
        record_product_interaction(session['user_id'], product_id, 'detail_view', 0.5)
    
    return jsonify(similar_products), 200

@app.route('/api/recommendations/popular', methods=['GET'])
def get_popular_products():
    limit = int(request.args.get('limit', 8))
    category = request.args.get('category')
    
    cur = mysql.connection.cursor()
    
    query = "SELECT * FROM products"
    params = []
    
    if category and category != 'all':
        query += " WHERE category = %s"
        params.append(category)
    
    query += " ORDER BY reviews_count DESC LIMIT %s"
    params.append(limit)
    
    cur.execute(query, params)
    products = cur.fetchall()
    cur.close()
    
    return jsonify(products), 200

# Helper Functions for Recommendation System
def record_product_interaction(user_id, product_id, interaction_type, weight):
    """Record user interactions with products"""
    try:
        cur = mysql.connection.cursor()
        cur.execute('''
        INSERT INTO user_interactions (user_id, product_id, interaction_type, weight)
        VALUES (%s, %s, %s, %s)
        ''', [user_id, product_id, interaction_type, weight])
        mysql.connection.commit()
        cur.close()
    except Exception as e:
        app.logger.error(f"Error recording interaction: {e}")

def record_search_interaction(user_id, search_term, category):
    """Record search interactions for better recommendations"""
    try:
        # Store search preferences
        if search_term:
            cur = mysql.connection.cursor()
            # Get existing search history for this user
            cur.execute('''
            SELECT preference_value FROM user_preferences
            WHERE user_id = %s AND preference_key = 'search_history'
            ''', [user_id])
            
            existing = cur.fetchone()
            if existing:
                # Update existing search history
                searches = json.loads(existing['preference_value'])
                # Add new search term if not already in the list
                if search_term.lower() not in [s.lower() for s in searches]:
                    searches.append(search_term)
                    # Keep only the latest 10 searches
                    if len(searches) > 10:
                        searches = searches[-10:]
                    
                cur.execute('''
                UPDATE user_preferences
                SET preference_value = %s
                WHERE user_id = %s AND preference_key = 'search_history'
                ''', [json.dumps(searches), user_id])
            else:
                # Create new search history
                cur.execute('''
                INSERT INTO user_preferences (user_id, preference_key, preference_value)
                VALUES (%s, %s, %s)
                ''', [user_id, 'search_history', json.dumps([search_term])])
            
            mysql.connection.commit()
            cur.close()
        
        # Store category preferences
        if category and category != 'all':
            cur = mysql.connection.cursor()
            # Get existing category preferences
            cur.execute('''
            SELECT preference_value FROM user_preferences
            WHERE user_id = %s AND preference_key = 'category_preferences'
            ''', [user_id])
            
            existing = cur.fetchone()
            if existing:
                # Update existing preferences
                categories = json.loads(existing['preference_value'])
                if category in categories:
                    # Increase count for this category
                    categories[category] += 1
                else:
                    # Add new category
                    categories[category] = 1
                
                cur.execute('''
                UPDATE user_preferences
                SET preference_value = %s
                WHERE user_id = %s AND preference_key = 'category_preferences'
                ''', [json.dumps(categories), user_id])
            else:
                # Create new category preferences
                cur.execute('''
                INSERT INTO user_preferences (user_id, preference_key, preference_value)
                VALUES (%s, %s, %s)
                ''', [user_id, 'category_preferences', json.dumps({category: 1})])
            
            mysql.connection.commit()
            cur.close()
    except Exception as e:
        app.logger.error(f"Error recording search interaction: {e}")

def get_content_based_recommendations(user_id, limit=8):
    """Get content-based recommendations for a user"""
    try:
        if not content_based_model:
            return []
        
        # Get user's interaction history
        cur = mysql.connection.cursor()
        cur.execute('''
        SELECT product_id, interaction_type, weight
        FROM user_interactions
        WHERE user_id = %s
        ORDER BY timestamp DESC
        ''', [user_id])
        interactions = cur.fetchall()
        
        # Get user preferences
        cur.execute('''
        SELECT preference_key, preference_value
        FROM user_preferences
        WHERE user_id = %s
        ''', [user_id])
        preferences = {row['preference_key']: json.loads(row['preference_value']) for row in cur.fetchall()}
        cur.close()
        
        if not interactions and not preferences:
            # If no interactions or preferences, return popular products
            return get_popular_products_from_db(limit)
        
        # Get user's favorite categories
        favorite_categories = []
        if 'category_preferences' in preferences:
            # Sort categories by preference count
            categories = preferences['category_preferences']
            favorite_categories = sorted(categories.keys(), key=lambda k: categories[k], reverse=True)
        
        # Get user's recent product interactions
        interacted_products = []
        for interaction in interactions:
            interacted_products.append({
                'product_id': interaction['product_id'],
                'weight': interaction['weight']
            })
        
        # Convert to DataFrame for processing
        df = content_based_model['products']
        tfidf_matrix = content_based_model['matrix']
        
        # Get product vectors for user's interacted products
        user_products = []
        for interaction in interacted_products[:10]:  # Consider only the 10 most recent
            product_idx = df[df['id'] == interaction['product_id']].index
            if len(product_idx) > 0:
                user_products.append((product_idx[0], interaction['weight']))
        
        # If user has interacted with products, find similar ones
        recommendations = []
        if user_products:
            # Create a user profile based on interacted products
            user_profile = np.zeros(tfidf_matrix.shape[1])
            weight_sum = 0
            
            for idx, weight in user_products:
                user_profile += tfidf_matrix[idx].toarray().flatten() * weight
                weight_sum += weight
            
            if weight_sum > 0:
                user_profile /= weight_sum
            
            # Calculate similarity scores
            sim_scores = cosine_similarity(user_profile.reshape(1, -1), tfidf_matrix).flatten()
            
            # Get indices of products sorted by similarity
            sim_indices = sim_scores.argsort()[::-1]
            
            # Filter out products the user has already interacted with
            interacted_ids = [p['product_id'] for p in interacted_products]
            sim_indices = [i for i in sim_indices if df.iloc[i]['id'] not in interacted_ids]
            
            # Get top recommendations
            top_indices = sim_indices[:limit]
            for idx in top_indices:
                product = df.iloc[idx].to_dict()
                recommendations.append(product)
        
        # If we don't have enough recommendations, add products from favorite categories
        if len(recommendations) < limit and favorite_categories:
            needed = limit - len(recommendations)
            existing_ids = [r['id'] for r in recommendations]
            
            for category in favorite_categories:
                if len(recommendations) >= limit:
                    break
                
                cur = mysql.connection.cursor()
                cur.execute('''
                SELECT * FROM products
                WHERE category = %s AND id NOT IN ({})
                ORDER BY rating DESC
                LIMIT %s
                '''.format(','.join(['%s'] * len(existing_ids))),
                [category] + existing_ids + [needed])
                
                category_products = cur.fetchall()
                cur.close()
                
                for product in category_products:
                    if product['id'] not in existing_ids:
                        recommendations.append(product)
                        existing_ids.append(product['id'])
                        if len(recommendations) >= limit:
                            break
        
        # If still not enough, add popular products
        if len(recommendations) < limit:
            needed = limit - len(recommendations)
            existing_ids = [r['id'] for r in recommendations]
            
            popular_products = get_popular_products_from_db(needed, existing_ids)
            recommendations.extend(popular_products)
        
        return recommendations[:limit]
    except Exception as e:
        app.logger.error(f"Error generating content-based recommendations: {e}")
        return []

def get_collaborative_recommendations(user_id, limit=8):
    """Get collaborative filtering recommendations for a user"""
    try:
        if not collaborative_model or user_id not in collaborative_model['user_map']:
            # If we don't have a collaborative model or user isn't in it, return empty list
            return []
        
        user_idx = collaborative_model['user_map'][user_id]
        user_features = collaborative_model['features'][user_idx].reshape(1, -1)
        
        # Calculate similarity with all other users
        similarities = cosine_similarity(user_features, collaborative_model['features'])
        similar_users_idx = similarities.argsort()[0][::-1][1:11]  # Top 10 similar users (excluding self)
        
        # Get similar users' IDs
        user_id_map = {i: uid for uid, i in collaborative_model['user_map'].items()}
        similar_users = [user_id_map[idx] for idx in similar_users_idx]
        
        # Get products that similar users have interacted with
        cur = mysql.connection.cursor()
        
        # Find products similar users interacted with highly
        placeholders = ','.join(['%s'] * len(similar_users))
        cur.execute(f'''
        SELECT ui.product_id, AVG(ui.weight) as avg_weight, COUNT(*) as user_count
        FROM user_interactions ui
        WHERE ui.user_id IN ({placeholders})
        GROUP BY ui.product_id
        ORDER BY avg_weight * user_count DESC
        ''', similar_users)
        
        similar_user_products = cur.fetchall()
        
        # Get products this user has already interacted with
        cur.execute('''
        SELECT product_id FROM user_interactions
        WHERE user_id = %s
        ''', [user_id])
        
        user_products = {row['product_id'] for row in cur.fetchall()}
        
        # Filter out products the user has already interacted with
        recommendations = []
        product_ids = []
        
        for row in similar_user_products:
            if row['product_id'] not in user_products and row['product_id'] not in product_ids:
                product_ids.append(row['product_id'])
                if len(product_ids) >= limit:
                    break
        
        # Get full product details
        if product_ids:
            placeholders = ','.join(['%s'] * len(product_ids))
            cur.execute(f'''
            SELECT * FROM products
            WHERE id IN ({placeholders})
            ''', product_ids)
            
            recommendations = cur.fetchall()
        
        cur.close()
        
        return recommendations
    except Exception as e:
        app.logger.error(f"Error generating collaborative recommendations: {e}")
        return []

def get_content_based_similar_products(product_id, limit=8):
    """Get content-based similar products"""
    try:
        if not content_based_model:
            return []
        
        df = content_based_model['products']
        tfidf_matrix = content_based_model['matrix']
        
        # Get index of the product
        product_idx = df[df['id'] == product_id].index
        if len(product_idx) == 0:
            return []
        
        product_idx = product_idx[0]
        
        # Calculate similarity scores
        sim_scores = cosine_similarity(tfidf_matrix[product_idx], tfidf_matrix).flatten()
        
        # Get indices of products sorted by similarity (excluding itself)
        sim_indices = list(sim_scores.argsort()[::-1])
        if product_idx in sim_indices:
            sim_indices.remove(product_idx)
        
        sim_indices = sim_indices[:limit]
        
        # Get similar products
        similar_products = []
        for idx in sim_indices:
            product = df.iloc[idx].to_dict()
            similar_products.append(product)
        
        return similar_products
    except Exception as e:
        app.logger.error(f"Error finding similar products: {e}")
        return []

def get_popular_products_from_db(limit, exclude_ids=None):
    """Get popular products from the database"""
    try:
        cur = mysql.connection.cursor()
        
        if exclude_ids and len(exclude_ids) > 0:
            placeholders = ','.join(['%s'] * len(exclude_ids))
            cur.execute(f'''
            SELECT * FROM products
            WHERE id NOT IN ({placeholders})
            ORDER BY reviews_count DESC
            LIMIT %s
            ''', exclude_ids + [limit])
        else:
            cur.execute('''
            SELECT * FROM products
            ORDER BY reviews_count DESC
            LIMIT %s
            ''', [limit])
        
        products = cur.fetchall()
        cur.close()
        
        return products
    except Exception as e:
        app.logger.error(f"Error getting popular products: {e}")
        return []

# Start the Flask application
if __name__ == '__main__':
    app.run(debug=True)