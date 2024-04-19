-- MariaDB dump 10.17  Distrib 10.4.13-MariaDB, for Linux (x86_64)
--
-- Host: cedar-mysql-vm    Database: functor_timeslice_4
-- ------------------------------------------------------
-- Server version	10.11.5-MariaDB-log

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;





CREATE DATABASE  IF NOT EXISTS `functor_timeslice_4` /*!40100 DEFAULT CHARACTER SET latin1 */;
USE `functor_timeslice_4`;
--
-- Table structure for table `cars`
--

DROP TABLE IF EXISTS `cars`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `cars` (
  `car_name` varchar(100) NOT NULL,
  PRIMARY KEY (`car_name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `cars`
--

LOCK TABLES `cars` WRITE;
/*!40000 ALTER TABLE `cars` DISABLE KEYS */;
INSERT INTO `cars` VALUES ('car_6304'),('car_6313'),('car_6316'),('car_6317'),('car_6322'),('car_6480'),('car_6512'),('car_6513'),('car_6520'),('car_6522'),('car_6545'),('car_6547'),('car_6666'),('ego_car_1');
/*!40000 ALTER TABLE `cars` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `ego_frame`
--

DROP TABLE IF EXISTS `ego_frame`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `ego_frame` (
  `f_id` int(11) NOT NULL,
  `ego_car_name` varchar(100) NOT NULL,
  `inLane` varchar(100) DEFAULT NULL,
  `inSpeed` int(11) DEFAULT NULL,
  PRIMARY KEY (`f_id`,`ego_car_name`),
  KEY `ego_car_name` (`ego_car_name`),
  CONSTRAINT `ego_frame_ibfk_1` FOREIGN KEY (`f_id`) REFERENCES `frames` (`f_id`),
  CONSTRAINT `ego_frame_ibfk_2` FOREIGN KEY (`ego_car_name`) REFERENCES `cars` (`car_name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `ego_frame`
--

LOCK TABLES `ego_frame` WRITE;
/*!40000 ALTER TABLE `ego_frame` DISABLE KEYS */;
INSERT INTO `ego_frame` VALUES (17840802,'ego_car_1','lane_middle',1),(17840804,'ego_car_1','lane_middle',1),(17840805,'ego_car_1','lane_middle',1),(17840806,'ego_car_1','lane_middle',1),(17840807,'ego_car_1','lane_middle',1),(17840808,'ego_car_1','lane_middle',1),(17840809,'ego_car_1','lane_middle',1),(17840810,'ego_car_1','lane_middle',1),(17840811,'ego_car_1','lane_middle',1),(17840812,'ego_car_1','lane_middle',1),(17840813,'ego_car_1','lane_middle',1),(17840814,'ego_car_1','lane_middle',1),(17840815,'ego_car_1','lane_middle',1),(17840816,'ego_car_1','lane_middle',1),(17840817,'ego_car_1','lane_middle',1),(17840818,'ego_car_1','lane_middle',1),(17840819,'ego_car_1','lane_middle',1),(17840820,'ego_car_1','lane_middle',2),(17840821,'ego_car_1','lane_middle',2),(17840822,'ego_car_1','lane_middle',2),(17840823,'ego_car_1','lane_middle',2),(17840824,'ego_car_1','lane_middle',2),(17840825,'ego_car_1','lane_middle',2),(17840826,'ego_car_1','lane_middle',2),(17840827,'ego_car_1','lane_middle',2),(17840828,'ego_car_1','lane_middle',2),(17840829,'ego_car_1','lane_middle',2),(17840830,'ego_car_1','lane_middle',2),(17840831,'ego_car_1','lane_middle',2),(17840832,'ego_car_1','lane_middle',2),(17840833,'ego_car_1','lane_middle',2),(17840834,'ego_car_1','lane_middle',2),(17840835,'ego_car_1','lane_middle',2),(17840836,'ego_car_1','lane_middle',2),(17840837,'ego_car_1','lane_middle',2),(17840838,'ego_car_1','lane_middle',2),(17840839,'ego_car_1','lane_middle',2),(17840840,'ego_car_1','lane_middle',2),(17840841,'ego_car_1','lane_middle',2),(17840842,'ego_car_1','lane_middle',2),(17840843,'ego_car_1','lane_middle',2),(17840844,'ego_car_1','lane_middle',2),(17840845,'ego_car_1','lane_middle',3),(17840846,'ego_car_1','lane_middle',3),(17840847,'ego_car_1','lane_middle',3),(17840848,'ego_car_1','lane_middle',3),(17840849,'ego_car_1','lane_middle',3),(17840850,'ego_car_1','lane_middle',3),(17840851,'ego_car_1','lane_middle',3),(17840852,'ego_car_1','lane_middle',3),(17840853,'ego_car_1','lane_middle',3),(17840854,'ego_car_1','lane_middle',3),(17840855,'ego_car_1','lane_middle',3),(17840856,'ego_car_1','lane_middle',3),(17840857,'ego_car_1','lane_middle',3),(17840858,'ego_car_1','lane_middle',4),(17840859,'ego_car_1','lane_middle',4),(17840860,'ego_car_1','lane_middle',4),(17840861,'ego_car_1','lane_middle',4),(17840862,'ego_car_1','lane_middle',4),(17840863,'ego_car_1','lane_middle',4),(17840864,'ego_car_1','lane_middle',4),(17840865,'ego_car_1','lane_middle',4),(17840866,'ego_car_1','lane_middle',4),(17840867,'ego_car_1','lane_middle',4),(17840868,'ego_car_1','lane_middle',4),(17840869,'ego_car_1','lane_middle',4),(17840870,'ego_car_1','lane_middle',4),(17840871,'ego_car_1','lane_middle',4),(17840872,'ego_car_1','lane_middle',4),(17840873,'ego_car_1','lane_middle',4),(17840874,'ego_car_1','lane_middle',4),(17840875,'ego_car_1','lane_middle',4),(17840876,'ego_car_1','lane_middle',4),(17840877,'ego_car_1','lane_middle',4),(17840878,'ego_car_1','lane_middle',4),(17840879,'ego_car_1','lane_middle',4),(17840880,'ego_car_1','lane_middle',4),(17840881,'ego_car_1','lane_middle',4),(17840882,'ego_car_1','lane_middle',4),(17840883,'ego_car_1','lane_middle',4),(17840884,'ego_car_1','lane_middle',4),(17840885,'ego_car_1','lane_middle',4),(17840886,'ego_car_1','lane_middle',4),(17840887,'ego_car_1','lane_middle',4),(17840888,'ego_car_1','lane_middle',4),(17840889,'ego_car_1','lane_middle',4),(17840890,'ego_car_1','lane_middle',4),(17840891,'ego_car_1','lane_middle',4),(17840892,'ego_car_1','lane_middle',4),(17840893,'ego_car_1','lane_middle',4),(17840894,'ego_car_1','lane_middle',4),(17840895,'ego_car_1','lane_middle',4),(17840896,'ego_car_1','lane_middle',4),(17840897,'ego_car_1','lane_middle',4),(17840898,'ego_car_1','lane_middle',4),(17840899,'ego_car_1','lane_middle',4),(17840900,'ego_car_1','lane_middle',4),(17840901,'ego_car_1','lane_middle',4),(17840902,'ego_car_1','lane_middle',4),(17840903,'ego_car_1','lane_middle',4),(17840904,'ego_car_1','lane_middle',4);
/*!40000 ALTER TABLE `ego_frame` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `frames`
--

DROP TABLE IF EXISTS `frames`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `frames` (
  `f_id` int(11) NOT NULL,
  PRIMARY KEY (`f_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `frames`
--

LOCK TABLES `frames` WRITE;
/*!40000 ALTER TABLE `frames` DISABLE KEYS */;
INSERT INTO `frames` VALUES (17840802),(17840804),(17840805),(17840806),(17840807),(17840808),(17840809),(17840810),(17840811),(17840812),(17840813),(17840814),(17840815),(17840816),(17840817),(17840818),(17840819),(17840820),(17840821),(17840822),(17840823),(17840824),(17840825),(17840826),(17840827),(17840828),(17840829),(17840830),(17840831),(17840832),(17840833),(17840834),(17840835),(17840836),(17840837),(17840838),(17840839),(17840840),(17840841),(17840842),(17840843),(17840844),(17840845),(17840846),(17840847),(17840848),(17840849),(17840850),(17840851),(17840852),(17840853),(17840854),(17840855),(17840856),(17840857),(17840858),(17840859),(17840860),(17840861),(17840862),(17840863),(17840864),(17840865),(17840866),(17840867),(17840868),(17840869),(17840870),(17840871),(17840872),(17840873),(17840874),(17840875),(17840876),(17840877),(17840878),(17840879),(17840880),(17840881),(17840882),(17840883),(17840884),(17840885),(17840886),(17840887),(17840888),(17840889),(17840890),(17840891),(17840892),(17840893),(17840894),(17840895),(17840896),(17840897),(17840898),(17840899),(17840900),(17840901),(17840902),(17840903),(17840904);
/*!40000 ALTER TABLE `frames` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `near`
--

DROP TABLE IF EXISTS `near`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `near` (
  `f_id` int(11) NOT NULL,
  `car_name` varchar(100) NOT NULL,
  `distance` int(11) DEFAULT NULL,
  `speed_diff` int(11) DEFAULT NULL,
  `near_level` int(11) DEFAULT NULL,
  `Lane` varchar(100) NOT NULL,
  `Speed` int(11) NOT NULL,
  PRIMARY KEY (`f_id`,`car_name`),
  KEY `car_name` (`car_name`),
  CONSTRAINT `near_ibfk_1` FOREIGN KEY (`f_id`) REFERENCES `frames` (`f_id`),
  CONSTRAINT `near_ibfk_2` FOREIGN KEY (`car_name`) REFERENCES `cars` (`car_name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `near`
--

LOCK TABLES `near` WRITE;
/*!40000 ALTER TABLE `near` DISABLE KEYS */;
INSERT INTO `near` VALUES (17840802,'car_6480',3,5,2,'lane_right',4),(17840804,'car_6480',2,5,2,'lane_right',4),(17840805,'car_6480',2,5,2,'lane_right',4),(17840806,'car_6480',2,5,2,'lane_right',4),(17840807,'car_6480',2,5,2,'lane_right',4),(17840808,'car_6480',2,5,2,'lane_right',4),(17840809,'car_6480',2,5,2,'lane_right',4),(17840810,'car_6480',2,5,2,'lane_right',4),(17840811,'car_6480',2,5,2,'lane_right',4),(17840812,'car_6480',2,5,2,'lane_right',4),(17840813,'car_6480',2,5,2,'lane_right',4),(17840814,'car_6480',2,5,2,'lane_right',4),(17840815,'car_6480',2,5,2,'lane_right',4),(17840816,'car_6480',2,5,2,'lane_right',4),(17840817,'car_6480',2,5,2,'lane_right',4),(17840818,'car_6480',2,5,2,'lane_right',4),(17840818,'car_6513',5,5,1,'lane_middle',4),(17840819,'car_6480',2,5,2,'lane_right',4),(17840819,'car_6513',5,5,1,'lane_middle',4),(17840820,'car_6480',2,5,2,'lane_right',4),(17840820,'car_6513',5,5,1,'lane_middle',4),(17840821,'car_6480',2,5,2,'lane_right',4),(17840821,'car_6513',5,5,1,'lane_middle',4),(17840822,'car_6313',5,5,1,'lane_right',5),(17840822,'car_6480',2,5,2,'lane_right',4),(17840822,'car_6513',5,5,1,'lane_middle',4),(17840823,'car_6313',5,5,1,'lane_right',5),(17840823,'car_6480',2,5,2,'lane_right',4),(17840823,'car_6513',5,5,1,'lane_middle',4),(17840824,'car_6313',5,5,1,'lane_right',5),(17840824,'car_6480',2,5,2,'lane_right',4),(17840824,'car_6513',5,5,1,'lane_middle',4),(17840825,'car_6313',5,5,1,'lane_right',5),(17840825,'car_6480',2,5,2,'lane_right',4),(17840825,'car_6513',5,5,1,'lane_middle',4),(17840826,'car_6313',5,5,1,'lane_right',5),(17840826,'car_6480',2,5,2,'lane_right',4),(17840826,'car_6513',5,5,1,'lane_middle',4),(17840827,'car_6313',5,5,1,'lane_right',5),(17840827,'car_6480',2,5,2,'lane_right',4),(17840827,'car_6513',5,5,1,'lane_middle',4),(17840828,'car_6313',5,5,1,'lane_right',5),(17840828,'car_6480',2,5,2,'lane_right',4),(17840828,'car_6513',5,5,1,'lane_middle',5),(17840829,'car_6313',5,5,1,'lane_right',5),(17840829,'car_6480',2,5,2,'lane_right',4),(17840829,'car_6513',4,5,1,'lane_middle',5),(17840830,'car_6313',5,5,1,'lane_right',5),(17840830,'car_6480',2,5,2,'lane_right',4),(17840830,'car_6513',4,5,1,'lane_middle',5),(17840831,'car_6313',5,5,1,'lane_right',5),(17840831,'car_6480',2,5,2,'lane_right',4),(17840831,'car_6513',4,5,1,'lane_middle',5),(17840832,'car_6313',5,5,1,'lane_right',5),(17840832,'car_6480',2,5,2,'lane_right',4),(17840832,'car_6513',4,5,1,'lane_middle',5),(17840833,'car_6313',5,5,1,'lane_right',5),(17840833,'car_6480',2,5,2,'lane_right',4),(17840833,'car_6513',4,5,1,'lane_middle',5),(17840834,'car_6313',4,5,1,'lane_right',5),(17840834,'car_6480',2,5,2,'lane_right',4),(17840834,'car_6513',4,5,1,'lane_middle',5),(17840835,'car_6313',4,5,1,'lane_right',5),(17840835,'car_6480',2,5,2,'lane_right',4),(17840835,'car_6513',4,5,1,'lane_middle',5),(17840836,'car_6313',4,5,1,'lane_right',5),(17840836,'car_6480',2,5,2,'lane_right',4),(17840836,'car_6513',4,5,1,'lane_middle',5),(17840837,'car_6313',4,5,1,'lane_right',5),(17840837,'car_6480',2,5,2,'lane_right',4),(17840837,'car_6513',4,5,1,'lane_middle',5),(17840838,'car_6313',4,5,1,'lane_right',5),(17840838,'car_6480',2,5,2,'lane_right',3),(17840838,'car_6513',4,5,1,'lane_middle',5),(17840839,'car_6313',4,5,1,'lane_right',5),(17840839,'car_6480',2,5,2,'lane_right',3),(17840839,'car_6513',4,5,1,'lane_middle',5),(17840840,'car_6313',4,5,1,'lane_right',5),(17840840,'car_6480',3,5,2,'lane_right',3),(17840840,'car_6513',3,5,1,'lane_middle',5),(17840841,'car_6313',4,5,1,'lane_right',5),(17840841,'car_6480',3,5,2,'lane_right',3),(17840841,'car_6513',3,5,2,'lane_middle',5),(17840842,'car_6313',3,5,1,'lane_right',5),(17840842,'car_6480',3,5,2,'lane_right',3),(17840842,'car_6513',3,5,2,'lane_middle',5),(17840843,'car_6313',4,5,1,'lane_right',5),(17840843,'car_6480',3,5,2,'lane_right',3),(17840843,'car_6513',3,5,2,'lane_middle',5),(17840844,'car_6313',3,5,1,'lane_right',5),(17840844,'car_6480',3,5,2,'lane_right',3),(17840844,'car_6513',3,5,2,'lane_middle',5),(17840845,'car_6313',3,5,1,'lane_right',5),(17840845,'car_6480',3,5,2,'lane_right',3),(17840845,'car_6513',3,5,2,'lane_middle',5),(17840846,'car_6313',3,5,1,'lane_right',5),(17840846,'car_6480',3,5,2,'lane_right',3),(17840846,'car_6513',3,5,2,'lane_middle',5),(17840847,'car_6313',3,5,1,'lane_right',5),(17840847,'car_6480',3,5,2,'lane_right',3),(17840847,'car_6513',3,5,2,'lane_middle',5),(17840848,'car_6313',3,5,2,'lane_right',5),(17840848,'car_6480',3,4,2,'lane_right',3),(17840848,'car_6513',3,5,2,'lane_middle',5),(17840849,'car_6313',3,5,2,'lane_right',5),(17840849,'car_6480',3,4,2,'lane_right',3),(17840849,'car_6513',3,5,2,'lane_middle',5),(17840850,'car_6313',3,5,2,'lane_right',5),(17840850,'car_6480',3,4,2,'lane_right',3),(17840850,'car_6513',3,5,2,'lane_middle',5),(17840851,'car_6313',3,5,2,'lane_right',5),(17840851,'car_6480',3,4,2,'lane_right',3),(17840851,'car_6513',3,5,2,'lane_middle',5),(17840852,'car_6313',3,5,2,'lane_right',5),(17840852,'car_6480',3,4,2,'lane_right',3),(17840852,'car_6513',2,5,2,'lane_middle',4),(17840853,'car_6313',3,5,2,'lane_right',5),(17840853,'car_6480',3,4,2,'lane_right',3),(17840853,'car_6513',3,5,2,'lane_middle',4),(17840854,'car_6313',3,5,2,'lane_right',5),(17840854,'car_6480',3,4,2,'lane_right',3),(17840854,'car_6513',2,5,2,'lane_middle',4),(17840855,'car_6313',3,5,2,'lane_right',5),(17840855,'car_6480',3,4,2,'lane_right',3),(17840855,'car_6513',2,5,2,'lane_middle',4),(17840856,'car_6313',3,5,2,'lane_right',5),(17840856,'car_6480',2,4,2,'lane_right',3),(17840856,'car_6513',2,4,2,'lane_middle',4),(17840857,'car_6313',3,5,2,'lane_right',5),(17840857,'car_6480',2,4,2,'lane_right',3),(17840857,'car_6513',2,4,2,'lane_middle',4),(17840858,'car_6313',3,5,2,'lane_right',5),(17840858,'car_6480',2,4,2,'lane_right',3),(17840858,'car_6513',2,4,2,'lane_middle',4),(17840859,'car_6313',3,5,2,'lane_right',5),(17840859,'car_6480',2,4,2,'lane_right',3),(17840859,'car_6513',2,4,2,'lane_middle',4),(17840860,'car_6313',3,5,2,'lane_right',5),(17840860,'car_6480',3,3,2,'lane_right',3),(17840860,'car_6513',2,4,2,'lane_middle',4),(17840861,'car_6313',3,5,2,'lane_right',5),(17840861,'car_6480',2,3,2,'lane_right',3),(17840861,'car_6513',2,4,2,'lane_middle',4),(17840862,'car_6313',3,4,2,'lane_right',5),(17840862,'car_6480',3,3,2,'lane_right',3),(17840862,'car_6513',2,3,2,'lane_middle',4),(17840863,'car_6313',3,4,2,'lane_right',5),(17840863,'car_6480',2,3,2,'lane_right',3),(17840863,'car_6513',2,3,2,'lane_middle',4),(17840864,'car_6313',3,4,2,'lane_right',5),(17840864,'car_6480',2,3,2,'lane_right',3),(17840864,'car_6513',2,3,2,'lane_middle',4),(17840865,'car_6313',3,4,2,'lane_right',5),(17840865,'car_6480',2,3,2,'lane_right',3),(17840865,'car_6513',2,3,2,'lane_middle',4),(17840866,'car_6313',3,4,2,'lane_right',5),(17840866,'car_6480',2,3,2,'lane_right',3),(17840866,'car_6513',2,3,2,'lane_middle',4),(17840867,'car_6313',3,4,2,'lane_right',5),(17840867,'car_6480',2,3,2,'lane_right',3),(17840867,'car_6513',2,3,2,'lane_middle',4),(17840868,'car_6313',3,4,2,'lane_right',4),(17840868,'car_6480',2,3,2,'lane_right',3),(17840868,'car_6513',2,3,2,'lane_middle',4),(17840869,'car_6313',3,4,2,'lane_right',4),(17840869,'car_6480',2,3,2,'lane_right',3),(17840869,'car_6513',2,3,2,'lane_middle',4),(17840870,'car_6313',3,4,2,'lane_right',4),(17840870,'car_6480',2,3,2,'lane_right',4),(17840870,'car_6513',2,3,2,'lane_middle',4),(17840870,'car_6666',5,5,1,'lane_right',5),(17840871,'car_6313',3,4,2,'lane_right',4),(17840871,'car_6480',2,3,2,'lane_right',4),(17840871,'car_6513',2,3,2,'lane_middle',4),(17840871,'car_6666',5,5,1,'lane_right',5),(17840872,'car_6313',3,3,2,'lane_right',4),(17840872,'car_6480',2,3,2,'lane_right',4),(17840872,'car_6513',2,3,2,'lane_middle',4),(17840872,'car_6666',5,5,1,'lane_right',5),(17840873,'car_6313',2,3,2,'lane_right',4),(17840873,'car_6480',2,3,2,'lane_right',4),(17840873,'car_6513',2,3,2,'lane_middle',4),(17840873,'car_6666',5,5,1,'lane_right',5),(17840874,'car_6313',2,3,2,'lane_right',3),(17840874,'car_6480',2,3,2,'lane_right',4),(17840874,'car_6513',2,3,2,'lane_middle',4),(17840874,'car_6666',5,5,1,'lane_right',5),(17840875,'car_6313',2,3,2,'lane_right',3),(17840875,'car_6480',2,3,2,'lane_right',4),(17840875,'car_6513',2,3,2,'lane_middle',4),(17840875,'car_6666',5,5,1,'lane_right',5),(17840876,'car_6313',3,2,2,'lane_right',3),(17840876,'car_6480',2,3,2,'lane_right',4),(17840876,'car_6513',2,3,2,'lane_middle',4),(17840876,'car_6666',5,4,1,'lane_right',5),(17840877,'car_6313',2,2,2,'lane_right',3),(17840877,'car_6480',2,3,2,'lane_right',4),(17840877,'car_6513',2,3,2,'lane_middle',4),(17840877,'car_6666',5,4,1,'lane_right',5),(17840878,'car_6313',3,2,2,'lane_right',3),(17840878,'car_6480',2,3,2,'lane_right',4),(17840878,'car_6513',2,3,2,'lane_middle',4),(17840878,'car_6666',5,4,1,'lane_right',5),(17840879,'car_6313',3,2,2,'lane_right',2),(17840879,'car_6480',2,3,2,'lane_right',4),(17840879,'car_6513',2,3,2,'lane_middle',4),(17840879,'car_6666',5,4,1,'lane_right',5),(17840880,'car_6313',3,2,2,'lane_right',2),(17840880,'car_6480',2,3,2,'lane_right',4),(17840880,'car_6513',3,3,2,'lane_middle',4),(17840880,'car_6666',5,4,1,'lane_right',5),(17840881,'car_6313',3,2,2,'lane_right',2),(17840881,'car_6480',2,3,2,'lane_right',4),(17840881,'car_6513',2,3,2,'lane_middle',4),(17840881,'car_6666',5,4,1,'lane_right',5),(17840882,'car_6313',3,1,2,'lane_right',2),(17840882,'car_6480',2,3,2,'lane_right',4),(17840882,'car_6513',3,3,2,'lane_middle',4),(17840882,'car_6666',5,4,1,'lane_right',5),(17840883,'car_6313',3,1,2,'lane_right',2),(17840883,'car_6480',2,3,2,'lane_right',4),(17840883,'car_6513',2,3,2,'lane_middle',4),(17840883,'car_6666',5,4,1,'lane_right',5),(17840884,'car_6313',3,1,2,'lane_right',1),(17840884,'car_6480',2,3,2,'lane_right',4),(17840884,'car_6513',3,3,2,'lane_middle',4),(17840884,'car_6666',5,4,1,'lane_right',5),(17840885,'car_6313',3,1,2,'lane_right',1),(17840885,'car_6480',2,3,2,'lane_right',4),(17840885,'car_6513',2,3,2,'lane_middle',4),(17840885,'car_6666',5,4,1,'lane_right',5),(17840886,'car_6313',3,1,2,'lane_right',1),(17840886,'car_6480',2,3,2,'lane_right',4),(17840886,'car_6513',3,3,2,'lane_middle',4),(17840886,'car_6666',5,4,1,'lane_right',5),(17840887,'car_6313',3,1,2,'lane_right',1),(17840887,'car_6480',2,4,2,'lane_right',4),(17840887,'car_6513',3,3,2,'lane_middle',3),(17840887,'car_6666',5,4,1,'lane_right',5),(17840888,'car_6313',3,1,2,'lane_right',1),(17840888,'car_6480',2,4,2,'lane_right',4),(17840888,'car_6513',3,2,2,'lane_middle',3),(17840888,'car_6666',5,4,1,'lane_right',5),(17840889,'car_6313',3,1,2,'lane_right',1),(17840889,'car_6480',2,4,2,'lane_right',4),(17840889,'car_6513',3,2,2,'lane_middle',3),(17840889,'car_6666',5,4,1,'lane_right',5),(17840890,'car_6313',3,1,1,'lane_right',1),(17840890,'car_6480',2,4,2,'lane_right',4),(17840890,'car_6513',3,2,2,'lane_middle',3),(17840890,'car_6666',5,4,1,'lane_right',5),(17840891,'car_6313',3,1,1,'lane_right',1),(17840891,'car_6480',2,4,2,'lane_right',4),(17840891,'car_6513',3,2,2,'lane_middle',2),(17840891,'car_6666',5,4,1,'lane_right',5),(17840892,'car_6313',4,1,1,'lane_right',1),(17840892,'car_6480',2,4,2,'lane_right',4),(17840892,'car_6513',3,2,2,'lane_middle',2),(17840892,'car_6666',5,4,1,'lane_right',5),(17840893,'car_6313',4,1,1,'lane_right',1),(17840893,'car_6480',2,4,2,'lane_right',4),(17840893,'car_6513',3,2,2,'lane_middle',2),(17840893,'car_6666',5,4,1,'lane_right',5),(17840894,'car_6313',4,1,1,'lane_right',1),(17840894,'car_6480',2,4,2,'lane_right',4),(17840894,'car_6513',3,1,2,'lane_middle',2),(17840894,'car_6666',5,4,1,'lane_right',5),(17840895,'car_6313',4,1,1,'lane_right',1),(17840895,'car_6480',2,4,2,'lane_right',4),(17840895,'car_6513',3,1,2,'lane_middle',1),(17840895,'car_6666',5,4,1,'lane_right',5),(17840896,'car_6313',4,1,1,'lane_right',1),(17840896,'car_6480',2,4,2,'lane_right',4),(17840896,'car_6513',3,1,2,'lane_middle',1),(17840896,'car_6666',5,4,1,'lane_right',5),(17840897,'car_6313',4,1,1,'lane_middle',1),(17840897,'car_6480',2,4,2,'lane_right',4),(17840897,'car_6513',3,1,2,'lane_middle',1),(17840897,'car_6666',5,4,1,'lane_right',5),(17840898,'car_6313',4,1,1,'lane_right',1),(17840898,'car_6480',2,4,2,'lane_right',4),(17840898,'car_6513',3,1,1,'lane_middle',1),(17840898,'car_6666',5,4,1,'lane_right',5),(17840899,'car_6313',4,1,1,'lane_right',1),(17840899,'car_6480',2,4,2,'lane_right',4),(17840899,'car_6513',3,1,1,'lane_middle',1),(17840899,'car_6666',5,4,1,'lane_right',5),(17840900,'car_6313',5,1,1,'lane_middle',1),(17840900,'car_6480',2,4,2,'lane_right',4),(17840900,'car_6513',4,1,1,'lane_middle',1),(17840900,'car_6666',5,4,1,'lane_right',5),(17840901,'car_6313',5,1,1,'lane_middle',1),(17840901,'car_6480',2,4,2,'lane_right',4),(17840901,'car_6513',4,1,1,'lane_middle',1),(17840901,'car_6666',5,4,1,'lane_right',5),(17840902,'car_6313',5,1,1,'lane_middle',1),(17840902,'car_6480',2,4,2,'lane_right',4),(17840902,'car_6513',4,1,1,'lane_middle',1),(17840902,'car_6666',5,4,1,'lane_right',5),(17840903,'car_6313',5,1,1,'lane_middle',1),(17840903,'car_6480',2,4,2,'lane_right',4),(17840903,'car_6513',4,1,1,'lane_middle',1),(17840903,'car_6666',5,4,1,'lane_right',5),(17840904,'car_6313',5,1,1,'lane_middle',1),(17840904,'car_6480',2,4,2,'lane_right',4),(17840904,'car_6513',4,1,1,'lane_middle',1),(17840904,'car_6666',5,4,1,'lane_right',5);
/*!40000 ALTER TABLE `near` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `succ_frame`
--

DROP TABLE IF EXISTS `succ_frame`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `succ_frame` (
  `f_id1` int(11) NOT NULL,
  `f_id2` int(11) NOT NULL,
  PRIMARY KEY (`f_id1`,`f_id2`),
  KEY `f_id2` (`f_id2`),
  CONSTRAINT `succ_frame_ibfk_1` FOREIGN KEY (`f_id1`) REFERENCES `frames` (`f_id`),
  CONSTRAINT `succ_frame_ibfk_2` FOREIGN KEY (`f_id2`) REFERENCES `frames` (`f_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `succ_frame`
--

LOCK TABLES `succ_frame` WRITE;
/*!40000 ALTER TABLE `succ_frame` DISABLE KEYS */;
INSERT INTO `succ_frame` VALUES (17840802,17840804),(17840804,17840805),(17840805,17840806),(17840806,17840807),(17840807,17840808),(17840808,17840809),(17840809,17840810),(17840810,17840811),(17840811,17840812),(17840812,17840813),(17840813,17840814),(17840814,17840815),(17840815,17840816),(17840816,17840817),(17840817,17840818),(17840818,17840819),(17840819,17840820),(17840820,17840821),(17840821,17840822),(17840822,17840823),(17840823,17840824),(17840824,17840825),(17840825,17840826),(17840826,17840827),(17840827,17840828),(17840828,17840829),(17840829,17840830),(17840830,17840831),(17840831,17840832),(17840832,17840833),(17840833,17840834),(17840834,17840835),(17840835,17840836),(17840836,17840837),(17840837,17840838),(17840838,17840839),(17840839,17840840),(17840840,17840841),(17840841,17840842),(17840842,17840843),(17840843,17840844),(17840844,17840845),(17840845,17840846),(17840846,17840847),(17840847,17840848),(17840848,17840849),(17840849,17840850),(17840850,17840851),(17840851,17840852),(17840852,17840853),(17840853,17840854),(17840854,17840855),(17840855,17840856),(17840856,17840857),(17840857,17840858),(17840858,17840859),(17840859,17840860),(17840860,17840861),(17840861,17840862),(17840862,17840863),(17840863,17840864),(17840864,17840865),(17840865,17840866),(17840866,17840867),(17840867,17840868),(17840868,17840869),(17840869,17840870),(17840870,17840871),(17840871,17840872),(17840872,17840873),(17840873,17840874),(17840874,17840875),(17840875,17840876),(17840876,17840877),(17840877,17840878),(17840878,17840879),(17840879,17840880),(17840880,17840881),(17840881,17840882),(17840882,17840883),(17840883,17840884),(17840884,17840885),(17840885,17840886),(17840886,17840887),(17840887,17840888),(17840888,17840889),(17840889,17840890),(17840890,17840891),(17840891,17840892),(17840892,17840893),(17840893,17840894),(17840894,17840895),(17840895,17840896),(17840896,17840897),(17840897,17840898),(17840898,17840899),(17840899,17840900),(17840900,17840901),(17840901,17840902),(17840902,17840903),(17840903,17840904);
/*!40000 ALTER TABLE `succ_frame` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2023-09-28 18:44:57
