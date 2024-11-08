/*==============================================================*/
/* Table: Customer                                              */
/*==============================================================*/
CREATE TABLE IF NOT EXISTS Customer (
   Id                   int primary key,
   FirstName            varchar(40)         not null,
   LastName             varchar(40)         not null,
   City                 varchar(40)         null,
   Country              varchar(40)         null,
   Phone                varchar(20)         null
);

INSERT INTO Customer (Id, FirstName, LastName, City, Country, Phone) 
VALUES (1, 'Maria', 'Anders', 'Berlin', 'Germany', '030-0074321'),
       (2, 'Ana', 'Trujillo', 'México D.F.', 'Mexico', '(5) 555-4729'),
       (3, 'Antonio', 'Moreno', 'México D.F.', 'Mexico', '(5) 555-3932');

ALTER TABLE Customer MODIFY Id int AUTO_INCREMENT;


/*==============================================================*/
/* Table: Orders                                                */
/*==============================================================*/
CREATE TABLE IF NOT EXISTS Orders (
   Id                   int primary key,
   OrderDate            datetime             not null default current_timestamp,
   OrderNumber          varchar(10)         null,
   CustomerId           int                  not null,
   TotalAmount          decimal(12,2)        null default 0
);

-- Insert rows with specific Ids as needed
-- INSERT INTO Orders (Id, OrderDate, OrderNumber, CustomerId, TotalAmount) VALUES (...);

ALTER TABLE Orders MODIFY Id int AUTO_INCREMENT;


/*==============================================================*/
/* Table: OrderItem                                             */
/*==============================================================*/
CREATE TABLE IF NOT EXISTS OrderItem (
   Id                   int primary key,
   OrderId              int                  not null,
   ProductId            int                  not null,
   UnitPrice            decimal(12,2)        not null default 0,
   Quantity             int                  not null default 1
);

-- Insert rows with specific Ids as needed
-- INSERT INTO OrderItem (Id, OrderId, ProductId, UnitPrice, Quantity) VALUES (...);

ALTER TABLE OrderItem MODIFY Id int AUTO_INCREMENT;


/*==============================================================*/
/* Table: Product                                               */
/*==============================================================*/
CREATE TABLE IF NOT EXISTS Product (
   Id                   int primary key,
   ProductName          varchar(50)         not null,
   SupplierId           int                  not null,
   UnitPrice            decimal(12,2)        null default 0,
   Package              varchar(30)         null,
   IsDiscontinued       tinyint(1)          not null default 0
);

-- Insert rows with specific Ids as needed
-- INSERT INTO Product (Id, ProductName, SupplierId, UnitPrice, Package, IsDiscontinued) VALUES (...);

ALTER TABLE Product MODIFY Id int AUTO_INCREMENT;


/*==============================================================*/
/* Table: Supplier                                              */
/*==============================================================*/
CREATE TABLE IF NOT EXISTS Supplier (
   Id                   int primary key,
   CompanyName          varchar(40)         not null,
   ContactName          varchar(50)         null,
   ContactTitle         varchar(40)         null,
   City                 varchar(40)         null,
   Country              varchar(40)         null,
   Phone                varchar(30)         null,
   Fax                  varchar(30)         null
);

-- Insert rows with specific Ids as needed
-- INSERT INTO Supplier (Id, CompanyName, ContactName, ContactTitle, City, Country, Phone, Fax) VALUES (...);

ALTER TABLE Supplier MODIFY Id int AUTO_INCREMENT;


/*==============================================================*/
/* Foreign Key Constraints                                      */
/*==============================================================*/
ALTER TABLE Orders
   ADD CONSTRAINT FK_ORDER_REFERENCE_CUSTOMER FOREIGN KEY (CustomerId)
      REFERENCES Customer (Id);

ALTER TABLE OrderItem
   ADD CONSTRAINT FK_ORDERITEM_REFERENCE_ORDER FOREIGN KEY (OrderId)
      REFERENCES Orders (Id);

ALTER TABLE OrderItem
   ADD CONSTRAINT FK_ORDERITEM_REFERENCE_PRODUCT FOREIGN KEY (ProductId)
      REFERENCES Product (Id);

ALTER TABLE Product
   ADD CONSTRAINT FK_PRODUCT_REFERENCE_SUPPLIER FOREIGN KEY (SupplierId)
      REFERENCES Supplier (Id);
