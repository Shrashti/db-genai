# Wanderbricks Data Schemas

This document contains the complete schema definitions for all tables in the Wanderbricks dataset. These schemas are used by the NL-to-SQL layer to generate accurate queries.

---

## Table: `wanderbricks.properties`

The main property listings table containing all vacation rental properties.

| Column | Type | Description | Constraints |
|--------|------|-------------|-------------|
| `property_id` | `bigint` | Unique identifier for the property | **Primary Key** |
| `host_id` | `bigint` | Reference to the property owner/host | Foreign Key → `hosts.host_id` |
| `city_id` | `bigint` | Reference to the city where property is located | Foreign Key → `cities.city_id` |
| `title` | `string` | Title of the property listing | |
| `description` | `string` | Detailed description of the property | |
| `base_price` | `float` | Base price per night for booking | |
| `property_type` | `string` | Type of property (e.g., house, apartment, villa) | |
| `max_guests` | `int` | Maximum number of guests allowed | |
| `bedrooms` | `int` | Number of bedrooms | |
| `bathrooms` | `int` | Number of bathrooms | |
| `created_at` | `date` | Date the property was listed | |

**Common Queries:**
- Search by location, price range, property type
- Filter by capacity (bedrooms, max_guests)
- Join with cities for location details

---

## Table: `wanderbricks.cities`

Reference table for city/location information.

| Column | Type | Description | Constraints |
|--------|------|-------------|-------------|
| `city_id` | `bigint` | Unique identifier for the city | **Primary Key** |
| `city` | `string` | Name of the city | |
| `country` | `string` | Name of the country | |
| `description` | `string` | Detailed description of the city (Markdown format) containing tourism info, demographics, history, geography, cuisine, and travel guidelines | |

**Common Queries:**
- Lookup city by name
- Search properties in a specific country
- Get city description for recommendations

---

## Table: `wanderbricks.hosts`

Information about property owners/hosts.

| Column | Type | Description | Constraints |
|--------|------|-------------|-------------|
| `host_id` | `bigint` | Unique identifier for the host | **Primary Key** |
| `name` | `string` | Host's full name | |
| `email` | `string` | Host's email address | |
| `phone` | `string` | Host's phone number | |
| `is_verified` | `boolean` | Whether the host is verified | |
| `is_active` | `boolean` | Whether the host account is active | |
| `rating` | `float` | Host rating (3.0-5.0 scale) | |
| `country` | `string` | Host's country of residence | |
| `joined_at` | `date` | Date the host joined the platform | |

**Common Queries:**
- Find top-rated hosts
- Filter verified/active hosts
- Get host details for a property

---

## Table: `wanderbricks.users`

Customer/guest information.

| Column | Type | Description | Constraints |
|--------|------|-------------|-------------|
| `user_id` | `bigint` | Unique identifier for the user | **Primary Key** |
| `email` | `string` | User's email address | |
| `name` | `string` | User's full name | |
| `country` | `string` | Country of residence | |
| `user_type` | `string` | Type of user (individual, business) | |
| `created_at` | `timestamp` | Account creation timestamp | |
| `is_business` | `boolean` | Whether the user is a business account | |
| `company_name` | `string` | Company name (if business user) | |

**Common Queries:**
- User lookup by email
- Filter business vs individual users
- Get booking history for a user

---

## Table: `wanderbricks.bookings`

Reservation/booking records.

| Column | Type | Description | Constraints |
|--------|------|-------------|-------------|
| `booking_id` | `bigint` | Unique identifier for the booking | **Primary Key** |
| `user_id` | `bigint` | Reference to the guest | Foreign Key → `users.user_id` |
| `property_id` | `bigint` | Reference to the booked property | Foreign Key → `properties.property_id` |
| `check_in` | `date` | Check-in date | |
| `check_out` | `date` | Check-out date | |
| `guests_count` | `int` | Number of guests for the booking | |
| `total_amount` | `float` | Total amount paid | |
| `status` | `string` | Booking status (pending, confirmed, cancelled) | |
| `created_at` | `date` | Date the booking was created | |
| `updated_at` | `date` | Date the booking was last updated | |

**Common Queries:**
- Get bookings for a property
- Find user's booking history
- Calculate revenue by property/host
- Check availability (date conflicts)

---

## Table: `wanderbricks.amenities`

Master list of available amenities.

| Column | Type | Description | Constraints |
|--------|------|-------------|-------------|
| `amenity_id` | `bigint` | Unique identifier for the amenity | **Primary Key** |
| `name` | `string` | Name of the amenity (e.g., Wi-Fi, Pool) | |
| `category` | `string` | Category of the amenity (Basic, Luxury) | |
| `icon` | `string` | Icon representation of the amenity | |

**Common Queries:**
- List all amenities by category
- Get amenity details

---

## Table: `wanderbricks.property_amenities`

Junction table linking properties to their amenities.

| Column | Type | Description | Constraints |
|--------|------|-------------|-------------|
| `property_id` | `bigint` | Reference to the property | Foreign Key → `properties.property_id` |
| `amenity_id` | `bigint` | Reference to the amenity | Foreign Key → `amenities.amenity_id` |

**Common Queries:**
- Get all amenities for a property
- Find properties with specific amenities
- Count amenities per property

---

## Table: `wanderbricks.property_images`

Images associated with properties.

| Column | Type | Description | Constraints |
|--------|------|-------------|-------------|
| `image_id` | `bigint` | Unique identifier for the image | **Primary Key** |
| `property_id` | `bigint` | Reference to the property | Foreign Key → `properties.property_id` |
| `url` | `string` | URL of the image | |
| `sequence` | `int` | Order in which to display images | |
| `is_primary` | `boolean` | Whether this is the primary/featured image | |
| `uploaded_at` | `date` | Date the image was uploaded | |

**Common Queries:**
- Get primary image for a property
- Get all images for a property (ordered)

---

## Table: `wanderbricks.employees`

Staff associated with hosts (cleaners, chefs, etc.).

| Column | Type | Description | Constraints |
|--------|------|-------------|-------------|
| `employee_id` | `bigint` | Unique identifier for the employee | **Primary Key** |
| `host_id` | `bigint` | Reference to the employer host | Foreign Key → `hosts.host_id` |
| `name` | `string` | Employee's full name | |
| `role` | `string` | Employee's role (cleaner, chef, etc.) | |
| `email` | `string` | Employee's email address | |
| `phone` | `string` | Employee's phone number | |
| `country` | `string` | Country of employment | |
| `joined_at` | `date` | Date the employee joined | |
| `end_service_date` | `date` | Date the employee left (if applicable) | |
| `is_currently_employed` | `boolean` | Whether currently employed | |

**Common Queries:**
- Get employees for a host
- Filter by role
- Active vs inactive employees

---

## Schema Statistics

| Table | Approximate Purpose |
|-------|---------------------|
| `properties` | Core listing data |
| `cities` | Location reference |
| `hosts` | Property owners |
| `users` | Guests/customers |
| `bookings` | Reservations |
| `amenities` | Amenity master list |
| `property_amenities` | Property-Amenity mapping |
| `property_images` | Listing photos |
| `employees` | Host staff |

---

## Usage in NL-to-SQL

This schema document is loaded by the `SQLGenerator` to provide context for query generation. The LLM uses this information to:

1. **Identify relevant tables** for a given query
2. **Determine correct JOINs** based on foreign key relationships
3. **Select appropriate columns** for filtering and projection
4. **Apply correct data types** in WHERE clauses
